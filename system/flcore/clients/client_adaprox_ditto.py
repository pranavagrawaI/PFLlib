import torch
import math
from flcore.clients.clientditto import clientDitto


class ClientAdaProxDitto(clientDitto):
    """
    AdaProxDitto Client: Adaptive proximal regularization for Ditto.
    Minimal, end-to-end, stable implementation using:
    - Sampled Li(wᵗ) evaluation (cap 256 examples)
    - Positive log-ratio gap: g_pos = clamp(log(Li/Lg), 0, gap_tau)
    - μ governor: mu_t = (1-γ)*mu_prev + γ*(mu_base + α*g_pos)
    - Median-based server EMA for robust global loss tracking
    
    Key behavior:
    - Computes sampled Li(w_t) on global model BEFORE personalized training
    - Sets adaptive mu based on positive log-ratio gap at START of ptrain()
    - Applies mu only to personalized optimizer, leaving global training unchanged
    - Logs minimal CSV: round, client_id, Li, Lg, g_pos, mu
    
    Acceptance sanity checks (leave as comments for quick validation):
    - By round 10: personalized accuracy ≥ Ditto baseline ±0.5 pp
    - median(mu) ≥ 0.06, fraction of μ≈0 client-rounds < 10%
    - Overhead ≤ +5% vs Ditto on MNIST/FEMNIST
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # AdaProx minimal config surface
        self.args = args  # Store full args for easy access
        self.alpha_gain = getattr(args, 'alpha_gain', 0.8)
        self.mu_base = getattr(args, 'mu_base', 0.08)
        self.mu_min = getattr(args, 'mu_min', 0.05)
        self.mu_max = getattr(args, 'mu_max', 3.0)
        self.gap_tau = getattr(args, 'gap_tau', 0.7)
        self.mu_smooth_gamma = getattr(args, 'mu_smooth_gamma', 0.3)
        self.warmup_rounds = getattr(args, 'warmup_rounds', 2)
        self.loss_eval_sample_ratio = getattr(args, 'loss_eval_sample_ratio', 0.1)
        
        # Persistent state for μ governor
        self.mu_prev = self.mu_min  # Initialize to floor
        self.mu_current = self.mu_min
        
        # Server-provided state (set by server during send_models)
        self.server_lg = None
        self.current_round = -1
        
        # Client-reported state (read by server after ptrain/train)
        self.mean_loss_global = 0.0
        self.adaptive_mu_info = {}

    def _eval_loss_on_global_sampled(self, sample_ratio: float, cap: int = 256) -> float:
        """
        Evaluate loss on a random subset of the local train set.
        Use up to 'cap' examples (whichever is smaller: ratio subset or cap).
        No gradients. Model in eval() mode. Return mean loss.
        """
        self.model.eval()
        trainloader = self.load_train_data()
        
        # Build a small sampled buffer
        buf_x, buf_y, n = [], [], 0
        # Pull at most 'cap' items, approximating sample_ratio
        target = max(1, min(cap, int(sample_ratio * getattr(self, "train_samples", cap))))
        
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x0 = x[0].to(self.device)
                else:
                    x0 = x.to(self.device)
                y0 = y.to(self.device)
                
                # Reservoir-like simple take-until target
                for i in range(x0.size(0)):
                    if n < target:
                        buf_x.append(x0[i].unsqueeze(0))
                        buf_y.append(y0[i].unsqueeze(0))
                        n += 1
                    if n >= target:
                        break
                if n >= target:
                    break

            if n == 0:
                return 0.0

            X = torch.cat(buf_x, dim=0)
            Y = torch.cat(buf_y, dim=0)
            logits = self.model(X)
            loss = self.loss(logits, Y)
            return float(loss.item())

    def _eval_accuracy_sampled(self, model, sample_ratio: float, cap: int = 256) -> float:
        """
        Evaluate accuracy on a random subset of the local train set.
        Use up to 'cap' examples (whichever is smaller: ratio subset or cap).
        No gradients. Model in eval() mode. Return accuracy percentage.
        
        Args:
            model: The model to evaluate (self.model for global, self.model_per for personalized)
            sample_ratio: Fraction of training data to sample
            cap: Maximum number of examples to evaluate
        
        Returns:
            Accuracy as a percentage (0-100)
        """
        model.eval()
        trainloader = self.load_train_data()
        
        # Build a small sampled buffer
        buf_x, buf_y, n = [], [], 0
        target = max(1, min(cap, int(sample_ratio * getattr(self, "train_samples", cap))))
        
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x0 = x[0].to(self.device)
                else:
                    x0 = x.to(self.device)
                y0 = y.to(self.device)
                
                # Reservoir-like simple take-until target
                for i in range(x0.size(0)):
                    if n < target:
                        buf_x.append(x0[i].unsqueeze(0))
                        buf_y.append(y0[i].unsqueeze(0))
                        n += 1
                    if n >= target:
                        break
                if n >= target:
                    break

            if n == 0:
                return 0.0

            X = torch.cat(buf_x, dim=0)
            Y = torch.cat(buf_y, dim=0)
            logits = model(X)
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == Y).sum().item()
            accuracy = 100.0 * correct / n
            return float(accuracy)

    def _set_adaptive_mu(self):
        """
        Compute adaptive mu based on positive log-ratio gap with governor:
        g = log((Li + eps) / (Lg + eps))
        g_pos = clamp(g, 0, gap_tau)
        mu_star = mu_base + alpha_gain * g_pos
        mu_t = (1 - mu_smooth_gamma) * mu_prev + mu_smooth_gamma * mu_star
        mu_t = clamp(mu_t, mu_min, mu_max)
        
        During warmup: mu_t = mu_min
        """
        eps = 1e-8
        Li = float(self.mean_loss_global)
        Lg = self.server_lg

        # Step-by-step tracking for detailed logging
        step_log = {
            "step_0_inputs": {
                "Li": Li,
                "Lg": float(Lg) if (Lg is not None and math.isfinite(Lg)) else None,
                "mu_prev": float(self.mu_prev),
            }
        }

        # Positive log-ratio gap (if Lg missing/invalid, treat as 0 boost)
        if (Lg is None) or (not math.isfinite(Lg)) or (not math.isfinite(Li)):
            g_raw = None
            g_pos = 0.0
        else:
            g_raw = math.log((Li + eps) / (Lg + eps))
            g_pos = min(max(g_raw, 0.0), self.gap_tau)

        step_log["step_1_gap"] = {
            "g_raw": float(g_raw) if g_raw is not None else None,
            "g_pos": float(g_pos),
            "gap_tau": float(self.gap_tau),
        }

        # Compute target mu_star
        mu_star = self.mu_base + self.alpha_gain * g_pos
        step_log["step_2_target"] = {
            "mu_star": float(mu_star),
            "mu_base": float(self.mu_base),
            "alpha_gain": float(self.alpha_gain),
        }

        # Apply temporal smoothing
        mu_smoothed = (1.0 - self.mu_smooth_gamma) * self.mu_prev + self.mu_smooth_gamma * mu_star
        step_log["step_3_smoothing"] = {
            "mu_smoothed": float(mu_smoothed),
            "gamma": float(self.mu_smooth_gamma),
        }

        # Apply bounds
        mu_t = max(self.mu_min, min(self.mu_max, mu_smoothed))
        step_log["step_4_bounded"] = {
            "mu_t": float(mu_t),
            "mu_min": float(self.mu_min),
            "mu_max": float(self.mu_max),
            "clipped": (mu_smoothed != mu_t),
        }

        # Warmup override
        if self.current_round < self.warmup_rounds:
            mu_t = self.mu_min
            step_log["step_5_warmup"] = {
                "warmup_active": True,
                "mu_final": float(mu_t),
            }
        else:
            step_log["step_5_warmup"] = {
                "warmup_active": False,
                "mu_final": float(mu_t),
            }

        self.mu_prev = mu_t
        self.mu_current = mu_t
        self.mu = mu_t  # for compatibility with Ditto parent

        if hasattr(self, "optimizer_per") and hasattr(self.optimizer_per, "mu"):
            self.optimizer_per.mu = mu_t

        # Compute accuracy metrics for logging
        acc_global = self._eval_accuracy_sampled(self.model, self.loss_eval_sample_ratio, cap=256)
        acc_personalized = self._eval_accuracy_sampled(self.model_per, self.loss_eval_sample_ratio, cap=256)

        self.adaptive_mu_info = {
            "mu": float(mu_t),
            "loss_local": Li,
            "loss_global_ema": float(Lg) if (Lg is not None) else 0.0,
            "g_pos": float(g_pos),
            "acc_global": float(acc_global),
            "acc_personalized": float(acc_personalized),
            "mu_evolution": step_log,  # Add detailed step tracking
        }

    def _log_csv_minimal(self):
        """
        Log minimal CSV with one line per client per round:
        round, client_id, Li, Lg, g_pos, mu
        Plus detailed step-by-step mu evolution columns
        """
        import os
        import csv
        
        outdir = getattr(self.args, "results_save_path", "./results")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "adaprox_minimal.csv")
        write_header = not os.path.exists(path)
        
        # Extract step-by-step evolution from adaptive_mu_info
        mu_evo = self.adaptive_mu_info.get("mu_evolution", {})
        step_0 = mu_evo.get("step_0_inputs", {})
        step_1 = mu_evo.get("step_1_gap", {})
        step_2 = mu_evo.get("step_2_target", {})
        step_3 = mu_evo.get("step_3_smoothing", {})
        step_4 = mu_evo.get("step_4_bounded", {})
        step_5 = mu_evo.get("step_5_warmup", {})
        
        row = {
            "round": int(self.current_round),
            "client_id": int(self.id),
            "Li": float(self.mean_loss_global),
            "Lg": float(self.server_lg) if self.server_lg is not None else 0.0,
            "g_pos": float(self.adaptive_mu_info.get("g_pos", 0.0)),
            "mu_final": float(self.mu_current),
            # Accuracy metrics
            "acc_global": float(self.adaptive_mu_info.get("acc_global", 0.0)),
            "acc_personalized": float(self.adaptive_mu_info.get("acc_personalized", 0.0)),
            # Step-by-step evolution
            "mu_prev": step_0.get("mu_prev", 0.0),
            "g_raw": step_1.get("g_raw", 0.0) if step_1.get("g_raw") is not None else 0.0,
            "g_pos_clipped": step_1.get("g_pos", 0.0),
            "mu_star": step_2.get("mu_star", 0.0),
            "mu_smoothed": step_3.get("mu_smoothed", 0.0),
            "mu_bounded": step_4.get("mu_t", 0.0),
            "warmup_active": 1 if step_5.get("warmup_active", False) else 0,
        }
        
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def _log_mu_evolution_console(self):
        """
        Optional: Print detailed μ evolution to console for debugging.
        Enable by setting args.verbose_mu = True
        """
        if not getattr(self.args, "verbose_mu", False):
            return
        
        mu_evo = self.adaptive_mu_info.get("mu_evolution", {})
        print(f"\n[Client {self.id} | Round {self.current_round}] μ Evolution:")
        
        # Step 0: Inputs
        step_0 = mu_evo.get("step_0_inputs", {})
        print("  Step 0 - Inputs:")
        print(f"    Li = {step_0.get('Li', 0.0):.6f}")
        print(f"    Lg = {step_0.get('Lg', 'None')}")
        print(f"    μ_prev = {step_0.get('mu_prev', 0.0):.6f}")
        
        # Accuracy metrics
        acc_global = self.adaptive_mu_info.get("acc_global", 0.0)
        acc_personalized = self.adaptive_mu_info.get("acc_personalized", 0.0)
        print(f"    Acc_global = {acc_global:.2f}%")
        print(f"    Acc_personalized = {acc_personalized:.2f}%")
        
        # Step 1: Gap computation
        step_1 = mu_evo.get("step_1_gap", {})
        g_raw = step_1.get("g_raw")
        print("  Step 1 - Gap Computation:")
        print(f"    g_raw = log(Li/Lg) = {g_raw:.6f if g_raw is not None else 'N/A'}")
        print(f"    g_pos = clamp(g_raw, 0, τ={step_1.get('gap_tau', 0.0):.2f}) = {step_1.get('g_pos', 0.0):.6f}")
        
        # Step 2: Target mu
        step_2 = mu_evo.get("step_2_target", {})
        print("  Step 2 - Target μ*:")
        print("    μ* = μ_base + α·g_pos")
        print(f"       = {step_2.get('mu_base', 0.0):.4f} + {step_2.get('alpha_gain', 0.0):.2f}·{step_1.get('g_pos', 0.0):.6f}")
        print(f"       = {step_2.get('mu_star', 0.0):.6f}")
        
        # Step 3: Smoothing
        step_3 = mu_evo.get("step_3_smoothing", {})
        gamma = step_3.get("gamma", 0.0)
        print("  Step 3 - Temporal Smoothing:")
        print("    μ_smooth = (1-γ)·μ_prev + γ·μ*")
        print(f"             = {1-gamma:.2f}·{step_0.get('mu_prev', 0.0):.6f} + {gamma:.2f}·{step_2.get('mu_star', 0.0):.6f}")
        print(f"             = {step_3.get('mu_smoothed', 0.0):.6f}")
        
        # Step 4: Bounding
        step_4 = mu_evo.get("step_4_bounded", {})
        clipped = step_4.get("clipped", False)
        print(f"  Step 4 - Bounds [{step_4.get('mu_min', 0.0):.4f}, {step_4.get('mu_max', 0.0):.4f}]:")
        print(f"    μ_bounded = {step_4.get('mu_t', 0.0):.6f} {'(clipped)' if clipped else '(no change)'}")
        
        # Step 5: Warmup
        step_5 = mu_evo.get("step_5_warmup", {})
        if step_5.get("warmup_active", False):
            print("  Step 5 - Warmup Override:")
            print(f"    μ_final = μ_min = {step_5.get('mu_final', 0.0):.6f} (warmup active)")
        else:
            print("  Step 5 - Final:")
            print(f"    μ_final = {step_5.get('mu_final', 0.0):.6f}")
        print()

    def ptrain(self):
        """
        Personalized training with adaptive proximal regularization.
        
        Critical sequence:
        1. Measure Li(w_t) on global model BEFORE any personalized updates (sampled)
        2. Compute and set adaptive mu based on loss gap
        3. Run standard Ditto personalized training with updated mu
        4. Log minimal CSV and optional console output
        """
        # Step 1: Sampled Li(w_t) before personalization
        self.mean_loss_global = self._eval_loss_on_global_sampled(
            sample_ratio=self.loss_eval_sample_ratio,
            cap=256
        )
        
        # Step 2: Set adaptive mu for this round
        self._set_adaptive_mu()
        
        # Step 2.5: Optional verbose console logging
        self._log_mu_evolution_console()
        
        # Step 3: Run standard Ditto personalized updates with new mu
        result = super().ptrain()
        
        # Step 4: Log minimal CSV
        self._log_csv_minimal()
        
        return result

    # Global training remains unchanged - inherit from parent
    # def train(self): return super().train()
