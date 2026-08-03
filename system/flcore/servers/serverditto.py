import numpy as np
import time
import os
import csv
from flcore.clients.clientditto import clientDitto
from flcore.servers.serverbase import Server
from threading import Thread


class Ditto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDitto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        # Separate tracking for personalized model metrics
        self.rs_test_acc_per = []
        self.rs_train_loss_per = []

    def _log_server_metrics_csv(self, round_num, test_acc_global, train_loss_global, test_acc_per, train_loss_per, avg_mu):
        """
        Log server-level metrics to CSV for baseline Ditto:
        round, test_acc_global, train_loss_global, test_acc_personalized, train_loss_personalized, mu (fixed), time_cost
        """
        outdir = getattr(self.args, "results_save_path", "./results")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "ditto_server_metrics.csv")
        write_header = not os.path.exists(path)

        row = {
            "round": int(round_num),
            "test_acc_global": float(test_acc_global) if test_acc_global is not None else 0.0,
            "train_loss_global": float(train_loss_global) if train_loss_global is not None else 0.0,
            "test_acc_personalized": float(test_acc_per) if test_acc_per is not None else 0.0,
            "train_loss_personalized": float(train_loss_per) if train_loss_per is not None else 0.0,
            "mu": float(avg_mu) if avg_mu is not None else 0.0,
            "time_cost": float(self.Budget[-1]) if self.Budget else 0.0,
        }

        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def train(self):
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global models")
                self.evaluate()

            if i%self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate_personalized()

                # Log server metrics to CSV
                test_acc_global = self.rs_test_acc[-1] if self.rs_test_acc else None
                train_loss_global = self.rs_train_loss[-1] if self.rs_train_loss else None
                test_acc_per = self.rs_test_acc_per[-1] if self.rs_test_acc_per else None
                train_loss_per = self.rs_train_loss_per[-1] if self.rs_train_loss_per else None
                avg_mu = self.args.lamda  # Fixed lambda for Ditto
                self._log_server_metrics_csv(i, test_acc_global, train_loss_global, test_acc_per, train_loss_per, avg_mu)
                self.log_server_metrics_csv(
                    i,
                    global_accuracy=test_acc_global,
                    personalized_accuracy=test_acc_per,
                    global_train_loss=train_loss_global,
                    personalized_train_loss=train_loss_per,
                    coefficients=[avg_mu],
                    time_cost=self.Budget[-1] if self.Budget else 0.0,
                )

            for client in self.selected_clients:
                client.ptrain()
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        final_round = len(self.Budget)
        self.global_round = final_round
        print(f"\n-------------Final model evaluation: round {final_round}-------------")
        print("\nEvaluate global models")
        self.evaluate()
        print("\nEvaluate personalized models")
        self.evaluate_personalized()
        test_acc_global = self.rs_test_acc[-1] if self.rs_test_acc else None
        train_loss_global = self.rs_train_loss[-1] if self.rs_train_loss else None
        test_acc_per = self.rs_test_acc_per[-1] if self.rs_test_acc_per else None
        train_loss_per = self.rs_train_loss_per[-1] if self.rs_train_loss_per else None
        self._log_server_metrics_csv(final_round, test_acc_global, train_loss_global, test_acc_per, train_loss_per, self.args.lamda)
        self.log_server_metrics_csv(
            final_round,
            global_accuracy=test_acc_global,
            personalized_accuracy=test_acc_per,
            global_train_loss=train_loss_global,
            personalized_train_loss=train_loss_per,
            coefficients=[self.args.lamda],
            time_cost=self.Budget[-1] if self.Budget else 0.0,
        )

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]) if len(self.Budget) > 1 else (self.Budget[0] if self.Budget else 0.0))

        self.save_results()
        self.save_global_model()
        self.log_event({"type": "run_finished", "status": "ok", "rounds": final_round})

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDitto)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def test_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics_personalized()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate_personalized(self, acc=None, loss=None):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        # Store in personalized tracking arrays
        self.rs_test_acc_per.append(test_acc)
        self.rs_train_loss_per.append(train_loss)

        if acc is not None:
            acc.append(test_acc)

        if loss is not None:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        round_num = getattr(self, "global_round", len(self.rs_test_acc_per) - 1)
        self._log_client_eval_csv("personalized", round_num, stats[0], stats[1], stats[2], stats[3], stats_train[1], stats_train[2])
        self.log_event({"type": "eval", "scope": "personalized", "round": round_num, "test_acc": test_acc, "test_auc": test_auc, "train_loss": train_loss})