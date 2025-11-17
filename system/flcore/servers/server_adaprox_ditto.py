import time
import os
import csv
from flcore.clients.client_adaprox_ditto import ClientAdaProxDitto
from flcore.servers.serverditto import Ditto
from flcore.servers.serverbase import Server


class ServerAdaProxDitto(Ditto):
    """
    AdaProxDitto: Adaptive proximal algorithm for Ditto.
    Combines Ditto's personalization with adaptive lambda based on loss gap.
    """
    def __init__(self, args, times):
        # Call Server.__init__ directly to avoid Ditto's set_clients(clientDitto)
        # Then manually do what Ditto.__init__ does but with our client class
        Server.__init__(self, args, times)
        
        # Replicate Ditto's init but with ClientAdaProxDitto
        self.set_slow_clients()
        self.set_clients(ClientAdaProxDitto)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        self.Budget = []
        
        # AdaProx-specific states
        self.lg = None
        self.beta = getattr(args, 'ema_beta', 0.9)
        
        # Separate tracking for personalized model metrics (Ditto-specific)
        self.rs_test_acc_per = []
        self.rs_train_loss_per = []
        
        print(f"\n[AdaProxDitto] EMA beta: {self.beta}")
        print("[AdaProxDitto] Using adaptive proximal regularization for Ditto")

    def _log_server_metrics_csv(self, round_num, median_loss, test_acc_global, train_loss_global, test_acc_per, train_loss_per, avg_mu):
        """
        Log server-level metrics to CSV:
        round, median_client_loss, lg_ema, test_acc_global, train_loss_global, test_acc_personalized, train_loss_personalized, avg_mu, time_cost
        """
        outdir = getattr(self.args, "results_save_path", "./results")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "adaprox_ditto_server_metrics.csv")
        write_header = not os.path.exists(path)
        
        row = {
            "round": int(round_num),
            "median_client_loss": float(median_loss) if median_loss is not None else 0.0,
            "lg_ema": float(self.lg) if self.lg is not None else 0.0,
            "test_acc_global": float(test_acc_global) if test_acc_global is not None else 0.0,
            "train_loss_global": float(train_loss_global) if train_loss_global is not None else 0.0,
            "test_acc_personalized": float(test_acc_per) if test_acc_per is not None else 0.0,
            "train_loss_personalized": float(train_loss_per) if train_loss_per is not None else 0.0,
            "avg_mu": float(avg_mu) if avg_mu is not None else 0.0,
            "time_cost": float(self.Budget[-1]) if self.Budget else 0.0,
        }
        
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def send_models(self):
        """
        Override to send both global model and server's EMA loss (lg) to clients.
        """
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()
            
            # Send global model parameters (from parent)
            client.set_parameters(self.global_model)
            
            # Send EMA and round info for adaptive lambda computation
            client.server_lg = self.lg
            client.current_round = self.global_round

            client.send_time = time.time() - start_time

    def train(self):
        """
        Override to insert EMA update logic after client training.
        Replicates the train() loop from Ditto with added EMA logic.
        """
        for i in range(self.global_rounds + 1):
            self.global_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global models")
                self.evaluate()

            if i % self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate_personalized()

            # Clients run personalized training (ptrain) then global training
            for client in self.selected_clients:
                client.ptrain()
                client.train()

            # Collect models from clients
            self.receive_models()
            
            # === AdaProx: Update EMA of median global loss ===
            med = None
            avg_mu = None
            try:
                import numpy as np
                client_losses = [float(c.mean_loss_global) for c in self.selected_clients 
                                 if hasattr(c, "mean_loss_global")]
                
                if len(client_losses) > 0:
                    med = float(np.median(client_losses))
                    # Update EMA with median
                    if self.lg is None:
                        self.lg = med
                    else:
                        self.lg = self.beta * self.lg + (1.0 - self.beta) * med
                    
                    if i % self.eval_gap == 0:
                        print(f"[AdaProxDitto] Median client loss: {med:.4f}, EMA (Lg): {self.lg:.4f}")
                
                # Compute average mu across clients
                client_mus = [float(c.mu_current) for c in self.selected_clients 
                             if hasattr(c, "mu_current")]
                if len(client_mus) > 0:
                    avg_mu = float(np.mean(client_mus))
                    if i % self.eval_gap == 0:
                        print(f"[AdaProxDitto] Average mu: {avg_mu:.4f}")
            
            except Exception as e:
                print(f"[AdaProxDitto Warning] Error computing EMA: {e}")
            # === End AdaProx logic ===
            
            # Log server metrics to CSV (need to track both global and personalized metrics for Ditto)
            if i % self.eval_gap == 0:
                # For Ditto, we have separate test accuracy tracking
                # rs_test_acc tracks global model accuracy
                # We need to manually track personalized accuracy from evaluate_personalized
                test_acc_global = self.rs_test_acc[-1] if self.rs_test_acc else None
                train_loss_global = self.rs_train_loss[-1] if self.rs_train_loss else None
                # Note: Ditto's evaluate_personalized adds to same rs_test_acc, so we'd need separate tracking
                # For now, we'll use the last entry which should be personalized if called last
                test_acc_per = None
                train_loss_per = None
                # Try to extract personalized metrics if available
                if hasattr(self, 'rs_test_acc_per'):
                    test_acc_per = self.rs_test_acc_per[-1] if self.rs_test_acc_per else None
                if hasattr(self, 'rs_train_loss_per'):
                    train_loss_per = self.rs_train_loss_per[-1] if self.rs_train_loss_per else None
                    
                self._log_server_metrics_csv(i, med, test_acc_global, train_loss_global, test_acc_per, train_loss_per, avg_mu)

            # DLG evaluation if needed
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            
            # Aggregate parameters (from parent)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(ClientAdaProxDitto)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def evaluate_personalized(self, acc=None, loss=None):
        """
        Override to track personalized model metrics in separate arrays.
        """
        # Call parent implementation but track in our own arrays
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        
        # Store in personalized tracking arrays
        self.rs_test_acc_per.append(test_acc)
        self.rs_train_loss_per.append(train_loss)
        
        # Also call parent to maintain compatibility
        super().evaluate_personalized(acc, loss)
