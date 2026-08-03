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
        self.shared_mu_current = getattr(args, 'mu_min', 0.05)
        self.track_overhead = getattr(args, 'track_overhead', False)

        print(f"\n[AdaProxFedProx] EMA beta: {self.beta}")
        print("[AdaProxFedProx] Using adaptive proximal regularization")

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and client  s.")

        # self.load_model()
        self.Budget = []


    def _reference_loss(self, client_losses):
        import numpy as np
        if not client_losses:
            return None
        if getattr(self.args, "controller_mode", "full") == "mean_global_loss":
            return float(np.mean(client_losses))
        return float(np.median(client_losses))

    def _update_global_loss_reference(self, reference_loss):
        if reference_loss is None:
            return
        if self.lg is None or getattr(self.args, "controller_mode", "full") == "no_ema":
            self.lg = reference_loss
        else:
            self.lg = self.beta * self.lg + (1.0 - self.beta) * reference_loss

    def _compute_shared_mu_for_next_round(self, reference_loss, previous_lg):
        import math
        mode = getattr(self.args, "controller_mode", "full")
        if mode != "global_shared":
            return
        if reference_loss is None or previous_lg is None or previous_lg <= 0:
            self.shared_mu_current = float(self.args.mu_min)
            return
        eps = 1e-8
        g_raw = math.log((float(reference_loss) + eps) / (float(previous_lg) + eps))
        g_pos = min(max(g_raw, 0.0), float(self.args.gap_tau))
        mu_star = float(self.args.mu_base) + float(self.args.alpha_gain) * g_pos
        previous = float(getattr(self, "shared_mu_current", self.args.mu_min))
        smoothed = (1.0 - float(self.args.mu_smooth_gamma)) * previous + float(self.args.mu_smooth_gamma) * mu_star
        self.shared_mu_current = max(float(self.args.mu_min), min(float(self.args.mu_max), smoothed))

    def _controller_summary(self, clients):
        import numpy as np
        coeffs = []
        deltas = []
        clipped = []
        for client in clients:
            info = getattr(client, "adaptive_mu_info", {})
            if not info:
                continue
            coeffs.append(float(info.get("mu", getattr(client, "mu_current", 0.0))))
            deltas.append(float(info.get("abs_mu_delta", 0.0)))
            clipped.append(1 if info.get("gap_clipped", False) or info.get("bound_clipped", False) else 0)
        return {
            "coefficients": coeffs,
            "avg_mu": float(np.mean(coeffs)) if coeffs else None,
            "mean_abs_coefficient_delta": float(np.mean(deltas)) if deltas else 0.0,
            "clipping_frequency": float(np.mean(clipped)) if clipped else 0.0,
        }

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

        # Send the global model to ALL clients (matching base send_models). The
        # previous selected-only version left non-selected clients holding stale
        # locally-trained models, which contaminated the global-accuracy metric
        # under partial participation (evaluate() scores all clients). Controller
        # state (lg / shared_mu / round) is only consumed by clients that train
        # this round, so it still goes to the selected clients only.
        for client in self.clients:
            client.set_parameters(self.global_model)

        for client in self.selected_clients:
            start_time = time.time()

            # Send EMA and round info for adaptive mu computation
            client.server_lg = self.lg
            client.server_shared_mu = getattr(self, "shared_mu_current", None)
            client.current_round = self.global_round

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def train(self):
        """
        Override to insert EMA update logic after client training.
        """
        for i in range(self.global_rounds):
            self.global_round = i
            s_t = time.time()
            t_eval = t_compute = t_controller = t_logging = t_agg = 0.0
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                _e = time.perf_counter()
                self.evaluate()
                t_eval = time.perf_counter() - _e

            # Clients run train() with adaptive mu
            _c = time.perf_counter()
            for client in self.selected_clients:
                client.train()
            t_compute = time.perf_counter() - _c

            # Collect models from clients
            _a = time.perf_counter()
            self.receive_models()
            t_agg += time.perf_counter() - _a

            # === AdaProx: Update EMA of median global loss ===
            med = None
            avg_mu = None
            summary = {"coefficients": [], "mean_abs_coefficient_delta": 0.0, "clipping_frequency": 0.0}
            _ctrl = time.perf_counter()
            try:
                import numpy as np
                client_losses = [float(c.mean_loss_global) for c in self.selected_clients
                                 if hasattr(c, "mean_loss_global")]
                old_lg = self.lg
                med = self._reference_loss(client_losses)
                self._update_global_loss_reference(med)
                self._compute_shared_mu_for_next_round(med, old_lg)
                summary = self._controller_summary(self.selected_clients)
                avg_mu = summary["avg_mu"]

                if med is not None and i % self.eval_gap == 0:
                    print(f"[AdaProxFedProx] Reference client loss: {med:.4f}, EMA (Lg): {self.lg:.4f}")
                if avg_mu is not None and i % self.eval_gap == 0:
                    print(f"[AdaProxFedProx] Average mu: {avg_mu:.4f}")

            except Exception as e:
                print(f"[AdaProxFedProx Warning] Error computing EMA: {e}")
            t_controller = time.perf_counter() - _ctrl
            # === End AdaProx logic ===

            # Log server metrics to CSV
            _l = time.perf_counter()
            if i % self.eval_gap == 0:
                test_acc = self.rs_test_acc[-1] if self.rs_test_acc else None
                train_loss = self.rs_train_loss[-1] if self.rs_train_loss else None
                self._log_server_metrics_csv(i, med, test_acc, train_loss, avg_mu)
                self.log_server_metrics_csv(
                    i,
                    global_accuracy=test_acc,
                    global_train_loss=train_loss,
                    reference_loss=med,
                    global_loss_ema=self.lg,
                    coefficients=summary.get("coefficients", []),
                    mean_abs_coefficient_delta=summary.get("mean_abs_coefficient_delta", 0.0),
                    clipping_frequency=summary.get("clipping_frequency", 0.0),
                    time_cost=self.Budget[-1] if self.Budget else 0.0,
                )
            t_logging = time.perf_counter() - _l

            # DLG evaluation if needed
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # Aggregate parameters (from parent)
            _a = time.perf_counter()
            self.aggregate_parameters()
            t_agg += time.perf_counter() - _a

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.track_overhead:
                t_probe = sum(getattr(c, "t_probe", 0.0) for c in self.selected_clients)
                t_ctrl_client = sum(getattr(c, "t_controller", 0.0) for c in self.selected_clients)
                self.log_overhead_csv(
                    i,
                    local_training_time=max(0.0, t_compute - t_probe - t_ctrl_client),
                    probe_loss_time=t_probe,
                    aggregation_time=t_agg,
                    controller_update_time=t_ctrl_client + t_controller,
                    evaluation_time=t_eval,
                    logging_time=t_logging,
                    total_round_time=self.Budget[-1],
                )

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        final_round = len(self.Budget)
        self.global_round = final_round
        print(f"\n-------------Final model evaluation: round {final_round}-------------")
        self.evaluate()
        summary = {"coefficients": [], "mean_abs_coefficient_delta": 0.0, "clipping_frequency": 0.0}
        try:
            import numpy as np
            client_losses = [float(c.mean_loss_global) for c in self.clients if hasattr(c, "mean_loss_global")]
            med = self._reference_loss(client_losses)
            summary = self._controller_summary(self.clients)
            avg_mu = summary["avg_mu"]
        except Exception:
            med = None
            avg_mu = None
        test_acc = self.rs_test_acc[-1] if self.rs_test_acc else None
        train_loss = self.rs_train_loss[-1] if self.rs_train_loss else None
        self._log_server_metrics_csv(final_round, med, test_acc, train_loss, avg_mu)
        self.log_server_metrics_csv(
            final_round,
            global_accuracy=test_acc,
            global_train_loss=train_loss,
            reference_loss=med,
            global_loss_ema=self.lg,
            coefficients=summary.get("coefficients", []),
            mean_abs_coefficient_delta=summary.get("mean_abs_coefficient_delta", 0.0),
            clipping_frequency=summary.get("clipping_frequency", 0.0),
            time_cost=self.Budget[-1] if self.Budget else 0.0,
        )

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]) if len(self.Budget) > 1 else (self.Budget[0] if self.Budget else 0.0))

        self.save_results()
        self.save_global_model()
        self.save_full_checkpoint()
        self.log_event({"type": "run_finished", "status": "ok", "rounds": final_round})

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAdaProx)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
