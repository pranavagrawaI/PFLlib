import time
import os
import csv
from flcore.clients.clientprox import clientProx
from flcore.servers.serverbase import Server
from threading import Thread


class FedProx(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProx)


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def _log_server_metrics_csv(self, round_num, test_acc, train_loss, avg_mu):
        """
        Log server-level metrics to CSV for baseline FedProx:
        round, test_acc, train_loss, mu (fixed), time_cost
        """
        outdir = getattr(self.args, "results_save_path", "./results")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "fedprox_server_metrics.csv")
        write_header = not os.path.exists(path)
        
        row = {
            "round": int(round_num),
            "test_acc": float(test_acc) if test_acc is not None else 0.0,
            "train_loss": float(train_loss) if train_loss is not None else 0.0,
            "mu": float(avg_mu) if avg_mu is not None else 0.0,
            "time_cost": float(self.Budget[-1]) if self.Budget else 0.0,
        }
        
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # Log server metrics to CSV
                test_acc = self.rs_test_acc[-1] if self.rs_test_acc else None
                train_loss = self.rs_train_loss[-1] if self.rs_train_loss else None
                avg_mu = self.args.mu  # Fixed mu for FedProx
                self._log_server_metrics_csv(i, test_acc, train_loss, avg_mu)

            for client in self.selected_clients:
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

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientProx)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
