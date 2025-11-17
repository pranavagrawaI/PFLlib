import time
import os
import csv
from flcore.clients.clientadaprox import clientAdaProx
from flcore.servers.serverbase import Server


class AdaProxFedProx(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        # Set custom client class (AdaProx clients instead of regular Prox clients)
        self.set_clients(clientAdaProx)
        
        # Global loss EMA state
        self.lg = None
        self.beta = getattr(args, 'ema_beta', 0.9)
        
        print(f"\n[AdaProxFedProx] EMA beta: {self.beta}")
        print("[AdaProxFedProx] Using adaptive proximal regularization")

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and client  s.")

        # self.load_model()
        self.Budget = []

    def _log_server_metrics_csv(self, round_num, median_loss, test_acc, train_loss, avg_mu):
        """
        Log server-level metrics to CSV:
        round, median_client_loss, lg_ema, test_acc, train_loss, avg_mu, time_cost
        """
        outdir = getattr(self.args, "results_save_path", "./results")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "adaprox_server_metrics.csv")
        write_header = not os.path.exists(path)
        
        row = {
            "round": int(round_num),
            "median_client_loss": float(median_loss) if median_loss is not None else 0.0,
            "lg_ema": float(self.lg) if self.lg is not None else 0.0,
            "test_acc": float(test_acc) if test_acc is not None else 0.0,
            "train_loss": float(train_loss) if train_loss is not None else 0.0,
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
            
            # Send EMA and round info for adaptive mu computation
            client.server_lg = self.lg
            client.current_round = self.global_round

            client.send_time = time.time() - start_time

    def train(self):
        """
        Override to insert EMA update logic after client training.
        """
        for i in range(self.global_rounds + 1):
            self.global_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # Clients run train() with adaptive mu
            for client in self.selected_clients:
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
                        print(f"[AdaProxFedProx] Median client loss: {med:.4f}, EMA (Lg): {self.lg:.4f}")
                
                # Compute average mu across clients
                client_mus = [float(c.mu_current) for c in self.selected_clients 
                             if hasattr(c, "mu_current")]
                if len(client_mus) > 0:
                    avg_mu = float(np.mean(client_mus))
                    if i % self.eval_gap == 0:
                        print(f"[AdaProxFedProx] Average mu: {avg_mu:.4f}")
            
            except Exception as e:
                print(f"[AdaProxFedProx Warning] Error computing EMA: {e}")
            # === End AdaProx logic ===
            
            # Log server metrics to CSV
            if i % self.eval_gap == 0:
                test_acc = self.rs_test_acc[-1] if self.rs_test_acc else None
                train_loss = self.rs_train_loss[-1] if self.rs_train_loss else None
                self._log_server_metrics_csv(i, med, test_acc, train_loss, avg_mu)

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
            self.set_new_clients(clientAdaProx)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
