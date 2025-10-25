import torch
import math
from flcore.clients.clientditto import clientDitto


class ClientAdaProxDitto(clientDitto):
    """
    AdaProxDitto Client: Adaptive proximal regularization for Ditto.
    Adaptively adjusts mu (proximal penalty) based on loss gap between
    client's global model loss and server's EMA of global losses.
    
    Key behavior:
    - Computes Li(w_t) on global model BEFORE personalized training
    - Sets adaptive mu = alpha * clip(Li(w_t) - Lg, 0, tau) at START of ptrain()
    - Applies mu only to personalized optimizer, leaving global training unchanged
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # AdaProx hyperparameters
        self.alpha = getattr(args, 'alpha_gain', 1.0)
        self.tau = getattr(args, 'gap_clip', 1.0)
        self.lam_max = getattr(args, 'lam_max', 5.0)
        self.warmup = getattr(args, 'warmup_rounds', 5)
        self.lam_init = getattr(args, 'lam_init', 0.0)
        
        # Server-provided state (set by server during send_models)
        self.server_lg = None
        self.current_round = -1
        
        # Client-reported state (read by server after ptrain/train)
        self.mean_loss_global = 0.0

    def _eval_loss_on_global_model(self) -> float:
        """
        Evaluate loss Li(w_t) of current global model on local training data.
        Used to compute the loss gap for adaptive mu.
        """
        trainloader = self.load_train_data()
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                total_loss += loss.item() * y.size(0)
                count += y.size(0)
        
        return total_loss / count if count > 0 else 0.0

    def _set_adaptive_mu(self):
        """
        Compute adaptive mu based on loss gap: mu(t) = alpha * clip(Li(w_t) - Lg, 0, tau).
        Capped at lam_max. Uses lam_init during warmup.
        
        Safety: Rejects non-finite inputs to prevent silent instability.
        """
        # Warmup, missing EMA, or non-finite values -> use init value
        lg = self.server_lg
        if (lg is None or 
            self.current_round < self.warmup or 
            not math.isfinite(lg) or 
            not math.isfinite(self.mean_loss_global)):
            lam = self.lam_init
            reason = "warmup" if self.current_round < self.warmup else "invalid_values"
        else:
            # Loss gap: how much worse is client's loss vs global EMA
            gap = float(self.mean_loss_global - float(lg))
            gap_clipped = max(0.0, min(self.tau, gap))  # clip to [0, tau]
            lam = min(self.lam_max, self.alpha * gap_clipped)  # scale and cap
            reason = f"gap={gap:.4f}"

        # Apply to personalized path only
        self.mu = lam
        if hasattr(self, "optimizer_per") and hasattr(self.optimizer_per, "mu"):
            self.optimizer_per.mu = lam  # PerturbedGradientDescent reads .mu in step()
        
        # Store for server-side logging
        self.adaptive_mu_info = {
            'mu': lam,
            'loss_local': self.mean_loss_global,
            'loss_global_ema': lg if lg is not None else 0.0,
            'reason': reason
        }

    def ptrain(self):
        """
        Personalized training with adaptive proximal regularization.
        
        Critical sequence:
        1. Measure Li(w_t) on global model BEFORE any personalized updates
        2. Compute and set adaptive mu based on loss gap
        3. Run standard Ditto personalized training with updated mu
        """
        # Step 1: Measure Li(w_t) before any updates
        self.mean_loss_global = self._eval_loss_on_global_model()
        
        # Step 2: Set adaptive mu for this round
        self._set_adaptive_mu()
        
        # Step 3: Run standard Ditto personalized updates with new mu
        return super().ptrain()

    # Global training remains unchanged - inherit from parent
    # def train(self): return super().train()
