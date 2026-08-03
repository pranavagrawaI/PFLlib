import torch
import math
import os
import csv
import time
from flcore.clients.clientprox import clientProx


class clientAdaProx(clientProx):
    """
    AdaProxFedProx Client: Adaptive proximal regularization for FedProx.
    Mirrors AdaProxDitto's mathematical structure with:
    - Sampled Li(wᵗ) evaluation (cap 256 examples)
    - Positive log-ratio gap: g_pos = clamp(log(Li/Lg), 0, gap_tau)
    - μ governor: mu_t = (1-γ)*mu_prev + γ*(mu_base + α*g_pos)
    - Median-based server EMA for robust global loss tracking
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Store args for easy access
        self.args = args

        # AdaProx hyperparameters - Ditto-style naming with backward compatibility
        self.alpha_gain = getattr(args, 'alpha_gain', getattr(args, 'alpha', 0.8))
        self.mu_base = getattr(args, 'mu_base', getattr(args, 'mu_init', 0.08))
        self.mu_min = getattr(args, 'mu_min', 0.05)
        self.mu_max = getattr(args, 'mu_max', 3.0)
        self.gap_tau = getattr(args, 'gap_tau', getattr(args, 'tau', 0.7))
        self.mu_smooth_gamma = getattr(args, 'mu_smooth_gamma', 0.3)
        self.warmup_rounds = getattr(args, 'warmup_rounds', getattr(args, 'warmup', 2))
        self.loss_eval_sample_ratio = getattr(args, 'loss_eval_sample_ratio', 0.1)

        # Persistent state for μ governor
        self.mu_prev = self.mu_min  # Initialize to floor
        self.mu_current = self.mu_min

        # Server-provided state (set by server during send_models)
        self.server_lg = None
        self.current_round = -1

        # Client-reported state (read by server after train)
        self.mean_loss_global = 0.0
        self.adaptive_mu_info = {}

        # Per-round timing (read by server for overhead_trace.csv)
        self.t_probe = 0.0
        self.t_controller = 0.0

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

    def _eval_accuracy_sampled(self, sample_ratio: float, cap: int = 256) -> float:
        """
        Evaluate accuracy on a random subset of the local train set.
        Use up to 'cap' examples (whichever is smaller: ratio subset or cap).
        No gradients. Model in eval() mode. Return accuracy percentage.

        Returns:
            Accuracy as a percentage (0-100)
        """
        self.model.eval()
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
            logits = self.model(X)
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == Y).sum().item()
            accuracy = 100.0 * correct / n
            return float(accuracy)

    def _set_adaptive_mu(self):
        """
        Compute the adaptive coefficient for this client/round.
        controller_mode toggles ablations while preserving the logged pipeline.
        """
        eps = 1e-8
        mode = getattr(self.args, "controller_mode", "full")
        Li = float(self.mean_loss_global)
        Lg = self.server_lg
        prev_mu = float(self.mu_prev)

        if (Lg is None) or (not math.isfinite(float(Lg))) or (not math.isfinite(Li)):
            g_raw = None
            g_unclipped = 0.0
        elif mode == "no_log_ratio":
            g_raw = (Li - float(Lg)) / (abs(float(Lg)) + eps)
            g_unclipped = max(g_raw, 0.0)
        else:
            g_raw = math.log((Li + eps) / (float(Lg) + eps))
            g_unclipped = max(g_raw, 0.0)

        if mode == "no_clipping":
            g_pos = g_unclipped
            gap_clipped = False
        else:
            g_pos = min(g_unclipped, self.gap_tau)
            gap_clipped = bool(g_unclipped > self.gap_tau)

        mu_star = self.mu_base + self.alpha_gain * g_pos
        mu_smoothed = (1.0 - self.mu_smooth_gamma) * prev_mu + self.mu_smooth_gamma * mu_star
        mu_bounded = max(self.mu_min, min(self.mu_max, mu_smoothed))
        bound_clipped = bool(mu_smoothed != mu_bounded)
        warmup_active = False

        if mode == "fixed_base":
            mu_t = float(self.mu_base)
            mu_star = mu_t
            mu_smoothed = mu_t
            mu_bounded = mu_t
            gap_clipped = False
            bound_clipped = False
        elif mode == "global_shared":
            shared_mu = getattr(self, "server_shared_mu", None)
            mu_t = float(shared_mu) if shared_mu is not None else float(self.mu_base)
            mu_star = mu_t
            mu_smoothed = mu_t
            mu_bounded = mu_t
            bound_clipped = False
        elif mode == "random_adaptive":
            span = max(0.0, float(self.mu_max) - float(self.mu_min))
            mu_t = float(self.mu_min) + span * float(torch.rand(1).item())
            mu_star = mu_t
            mu_smoothed = mu_t
            mu_bounded = mu_t
            gap_clipped = False
            bound_clipped = False
        else:
            if mode == "no_clipping":
                mu_t = mu_smoothed
            else:
                mu_t = mu_bounded
            if self.current_round < self.warmup_rounds:
                mu_t = self.mu_min
                warmup_active = True

        self.mu_prev = mu_t
        self.mu_current = mu_t
        self.mu = mu_t

        optimizer = getattr(self, "optimizer_per", getattr(self, "optimizer", None))
        if optimizer is not None:
            for group in optimizer.param_groups:
                group["mu"] = mu_t

        acc_global = 0.0
        acc_personalized = 0.0
        if getattr(self.args, "track_probe_accuracy", False):
            if hasattr(self, "model_per"):
                acc_global = self._eval_accuracy_sampled(self.model, self.loss_eval_sample_ratio, cap=256)
                acc_personalized = self._eval_accuracy_sampled(self.model_per, self.loss_eval_sample_ratio, cap=256)
            else:
                acc_global = self._eval_accuracy_sampled(self.loss_eval_sample_ratio, cap=256)

        step_log = {
            "step_0_inputs": {
                "Li": Li,
                "Lg": float(Lg) if Lg is not None else None,
                "mu_prev": prev_mu,
                "controller_mode": mode,
            },
            "step_1_gap": {
                "g_raw": float(g_raw) if g_raw is not None else None,
                "g_unclipped": float(g_unclipped),
                "g_pos": float(g_pos),
                "gap_tau": float(self.gap_tau),
                "gap_clipped": gap_clipped,
            },
            "step_2_target": {
                "mu_star": float(mu_star),
                "mu_base": float(self.mu_base),
                "alpha_gain": float(self.alpha_gain),
            },
            "step_3_smoothing": {
                "mu_smoothed": float(mu_smoothed),
                "gamma": float(self.mu_smooth_gamma),
            },
            "step_4_bounded": {
                "mu_t": float(mu_bounded),
                "mu_min": float(self.mu_min),
                "mu_max": float(self.mu_max),
                "clipped": bound_clipped,
            },
            "step_5_warmup": {
                "warmup_active": warmup_active,
                "mu_final": float(mu_t),
            },
        }

        self.adaptive_mu_info = {
            "controller_mode": mode,
            "mu": float(mu_t),
            "loss_local": Li,
            "loss_global_ema": float(Lg) if Lg is not None else 0.0,
            "g_raw": float(g_raw) if g_raw is not None else 0.0,
            "g_pos": float(g_pos),
            "gap_clipped": gap_clipped,
            "bound_clipped": bound_clipped,
            "warmup_active": warmup_active,
            "mu_prev": prev_mu,
            "mu_star": float(mu_star),
            "mu_smoothed": float(mu_smoothed),
            "mu_bounded": float(mu_bounded),
            "abs_mu_delta": abs(float(mu_t) - prev_mu),
            "acc_global": float(acc_global),
            "acc_personalized": float(acc_personalized),
            "mu_evolution": step_log,
        }

    def _log_csv_minimal(self):
        """
        Log one controller row per selected client/round.
        Also keeps the legacy minimal CSV filename for old analysis scripts.
        """
        outdir = getattr(self.args, "results_save_path", "./results")
        os.makedirs(outdir, exist_ok=True)
        info = self.adaptive_mu_info
        row = {
            "round": int(self.current_round),
            "client_id": int(self.id),
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "controller_mode": info.get("controller_mode", getattr(self.args, "controller_mode", "full")),
            "Li": float(self.mean_loss_global),
            "Lg": float(self.server_lg) if self.server_lg is not None else 0.0,
            "client_loss_gap_raw": float(info.get("g_raw", 0.0)),
            "client_loss_gap_clipped": float(info.get("g_pos", 0.0)),
            "gap_tau": float(self.gap_tau),
            "coefficient_prev": float(info.get("mu_prev", 0.0)),
            "coefficient_target": float(info.get("mu_star", 0.0)),
            "coefficient_smoothed": float(info.get("mu_smoothed", 0.0)),
            "coefficient_bounded": float(info.get("mu_bounded", 0.0)),
            "coefficient_final": float(self.mu_current),
            "gap_clipped": 1 if info.get("gap_clipped", False) else 0,
            "bound_clipped": 1 if info.get("bound_clipped", False) else 0,
            "warmup_active": 1 if info.get("warmup_active", False) else 0,
            "abs_coefficient_delta": float(info.get("abs_mu_delta", 0.0)),
            "acc_global": float(info.get("acc_global", 0.0)),
            "acc_personalized": float(info.get("acc_personalized", 0.0)),
        }
        fieldnames = list(row.keys())
        for filename in ("client_controller.csv", self._legacy_controller_filename()):
            path = os.path.join(outdir, filename)
            write_header = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                w.writerow(row)

    def _legacy_controller_filename(self):
        return "adaprox_ditto_minimal.csv" if hasattr(self, "model_per") else "adaprox_minimal.csv"

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
        print(f"    Acc_global = {acc_global:.2f}%")

        # Step 1: Gap computation
        step_1 = mu_evo.get("step_1_gap", {})
        g_raw = step_1.get("g_raw")
        print("  Step 1 - Gap Computation:")
        print(f"    g_raw = log(Li/Lg) = {g_raw:.6f}" if g_raw is not None else "    g_raw = log(Li/Lg) = N/A")
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



    def train(self):
        """
        Training with adaptive proximal regularization.

        Critical sequence (mirroring Ditto):
        1. Measure Li(w_t) on global model BEFORE any local updates (sampled)
        2. Compute and set adaptive mu based on loss gap
        3. Run standard FedProx training with updated mu
        4. Log minimal CSV and optional console output
        """
        # Step 1: Sampled Li(w_t) before training
        _t0 = time.perf_counter()
        self.mean_loss_global = self._eval_loss_on_global_sampled(
            sample_ratio=self.loss_eval_sample_ratio,
            cap=256
        )
        _t1 = time.perf_counter()

        # Step 2: Set adaptive mu for this round
        self._set_adaptive_mu()
        _t2 = time.perf_counter()

        # Step 2.5: Optional verbose console logging
        self._log_mu_evolution_console()

        # Step 3: Run standard FedProx training with new mu
        super().train()

        # Step 4: Log minimal CSV
        self._log_csv_minimal()

        self.t_probe = _t1 - _t0
        self.t_controller = _t2 - _t1
