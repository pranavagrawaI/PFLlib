import time
from flcore.clients.client_adaprox_ditto import ClientAdaProxDitto
from flcore.servers.serverditto import Ditto


class ServerAdaProxDitto(Ditto):
    """
    AdaProxDitto: Adaptive proximal algorithm for Ditto.
    Combines Ditto's personalization with adaptive lambda based on loss gap.
    """
    def __init__(self, args, times):
        # Call Server.__init__ directly to avoid Ditto's set_clients(clientDitto)
        # Then manually do what Ditto.__init__ does but with our client class
        from flcore.servers.serverbase import Server
        Server.__init__(self, args, times)
        
        # Replicate Ditto's init but with ClientAdaProxDitto
        self.set_slow_clients()
        self.set_clients(ClientAdaProxDitto)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        self.Budget = []
        
        # AdaProx-specific state
        self.lg = None
        self.beta = getattr(args, 'ema_beta', 0.9)
        
        print(f"\n[AdaProxDitto] EMA beta: {self.beta}")
        print("[AdaProxDitto] Using adaptive proximal regularization for Ditto")

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
            
            except Exception as e:
                print(f"[AdaProxDitto Warning] Error computing EMA: {e}")
            # === End AdaProx logic ===

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
