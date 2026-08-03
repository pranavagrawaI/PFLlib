import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import json
import csv
from utils.data_utils import read_client_data
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = max(1, int(self.num_clients * self.join_ratio))
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break
        self.times = times
        self.results_save_path = os.path.abspath(getattr(args, "results_save_path", "./results"))
        self.model_save_path = getattr(args, "model_save_path", None)
        if self.model_save_path:
            self.model_save_path = os.path.abspath(self.model_save_path)
            os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.results_save_path, exist_ok=True)
        self.event_log_path = os.path.join(self.results_save_path, "events.jsonl")

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args,
                            id=i,
                            train_samples=len(train_data),
                            test_samples=len(test_data),
                            train_slow=train_slow,
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_count = int((1 - self.client_drop_rate) * self.current_num_join_clients)
        active_count = max(1, min(len(self.selected_clients), active_count))
        active_clients = random.sample(self.selected_clients, active_count)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        candidates = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            candidates.append((client_time_cost, client))
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)

        if tot_samples <= 0 and candidates:
            client_time_cost, client = min(candidates, key=lambda item: item[0])
            print(f"[Warning] No clients passed time threshold; keeping fastest client {client.id} ({client_time_cost:.4f}s).")
            tot_samples = client.train_samples
            self.uploaded_ids = [client.id]
            self.uploaded_weights = [client.train_samples]
            self.uploaded_models = [client.model]

        if tot_samples <= 0:
            raise RuntimeError("No client updates available for aggregation.")

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = self.model_save_path or os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)
        self.log_event({"type": "model_saved", "path": model_path})

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = self.results_save_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = os.path.join(result_path, "{}.h5".format(algo))
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                if hasattr(self, 'rs_test_acc_per'):
                    hf.create_dataset('rs_test_acc_per', data=self.rs_test_acc_per)
                if hasattr(self, 'rs_train_loss_per'):
                    hf.create_dataset('rs_train_loss_per', data=self.rs_train_loss_per)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_samples = max(1, sum(stats[1]))
        train_samples = max(1, sum(stats_train[1]))
        test_acc = sum(stats[2])*1.0 / test_samples
        test_auc = sum(stats[3])*1.0 / test_samples
        train_loss = sum(stats_train[2])*1.0 / train_samples
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
            self.rs_test_auc.append(test_auc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        round_num = getattr(self, "global_round", len(self.rs_test_acc) - 1)
        self._log_client_eval_csv("global", round_num, stats[0], stats[1], stats[2], stats[3], stats_train[1], stats_train[2])
        self.log_event({"type": "eval", "scope": "global", "round": round_num, "test_acc": test_acc, "test_auc": test_auc, "train_loss": train_loss})

    def _write_csv_row(self, filename, row, fieldnames=None):
        path = os.path.join(self.results_save_path, filename)
        write_header = not os.path.exists(path)
        if fieldnames is None:
            fieldnames = list(row.keys())
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _log_client_eval_csv(self, scope, round_num, ids, test_samples, corrects, aucs, train_samples, train_losses):
        train_by_pos = {pos: (train_losses[pos], train_samples[pos]) for pos in range(len(train_samples))}
        for pos, cid in enumerate(ids):
            ts = max(1, int(test_samples[pos]))
            tr_loss, tr_samples = train_by_pos.get(pos, (0.0, 0))
            row = {
                "round": int(round_num),
                "scope": scope,
                "client_id": int(cid),
                "test_samples": int(test_samples[pos]),
                "test_accuracy": float(corrects[pos]) / ts,
                "test_auc": float(aucs[pos]) / ts,
                "train_samples": int(tr_samples),
                "train_loss": float(tr_loss) / max(1, int(tr_samples)),
            }
            self._write_csv_row("client_eval.csv", row)

    def log_server_metrics_csv(self, round_num, **metrics):
        coeffs = metrics.pop("coefficients", None)
        if coeffs is None:
            coeffs = []
        coeffs = [float(x) for x in coeffs if x is not None and np.isfinite(float(x))]
        selected_ids = [int(c.id) for c in getattr(self, "selected_clients", [])]
        row = {
            "round": int(round_num),
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "goal": self.goal,
            "seed": int(getattr(self.args, "current_seed", getattr(self.args, "seed", 0))),
            "controller_mode": getattr(self.args, "controller_mode", "full"),
            "selected_clients": len(selected_ids),
            "selected_client_ids": " ".join(map(str, selected_ids)),
            "global_accuracy": float(metrics.pop("global_accuracy", 0.0) or 0.0),
            "personalized_accuracy": float(metrics.pop("personalized_accuracy", 0.0) or 0.0),
            "global_train_loss": float(metrics.pop("global_train_loss", 0.0) or 0.0),
            "personalized_train_loss": float(metrics.pop("personalized_train_loss", 0.0) or 0.0),
            "reference_loss": float(metrics.pop("reference_loss", 0.0) or 0.0),
            "global_loss_ema": float(metrics.pop("global_loss_ema", 0.0) or 0.0),
            "coefficient_mean": float(np.mean(coeffs)) if coeffs else float(metrics.pop("coefficient_mean", 0.0) or 0.0),
            "coefficient_variance": float(np.var(coeffs)) if coeffs else float(metrics.pop("coefficient_variance", 0.0) or 0.0),
            "mean_abs_coefficient_delta": float(metrics.pop("mean_abs_coefficient_delta", 0.0) or 0.0),
            "clipping_frequency": float(metrics.pop("clipping_frequency", 0.0) or 0.0),
            "time_cost": float(metrics.pop("time_cost", 0.0) or 0.0),
            "unstable": int(metrics.pop("unstable", 0) or 0),
        }
        for key, value in metrics.items():
            row[key] = value
        self._write_csv_row("server_metrics.csv", row)

    def log_overhead_csv(self, round_num, *, local_training_time, probe_loss_time,
                         aggregation_time, controller_update_time, evaluation_time,
                         logging_time, total_round_time):
        """Per-round timing breakdown + GPU memory (MB). Written only when
        --track_overhead is enabled; lets the overhead study separate algorithmic
        cost from instrumentation cost."""
        gpu_alloc = gpu_res = 0.0
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / 1e6
            gpu_res = torch.cuda.memory_reserved() / 1e6
        row = {
            "round": int(round_num),
            "local_training_time": float(local_training_time),
            "probe_loss_time": float(probe_loss_time),
            "aggregation_time": float(aggregation_time),
            "controller_update_time": float(controller_update_time),
            "evaluation_time": float(evaluation_time),
            "logging_time": float(logging_time),
            "total_round_time": float(total_round_time),
            "gpu_memory_allocated": float(gpu_alloc),
            "gpu_memory_reserved": float(gpu_res),
        }
        self._write_csv_row("overhead_trace.csv", row)

    def save_full_checkpoint(self):
        """Full-state checkpoint so a run can be faithfully resumed/analyzed:
        global weights, per-client personalized weights (Ditto-like), per-client
        proximal coefficients, server loss reference, round, seed, and config."""
        path = self.model_save_path or os.path.join("models", self.dataset)
        os.makedirs(path, exist_ok=True)
        ckpt = {
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "round": int(getattr(self, "global_round", len(self.rs_test_acc))),
            "seed": int(getattr(self.args, "current_seed", getattr(self.args, "seed", 0))),
            "global_model_state_dict": self.global_model.state_dict(),
            "server_Lg_state": getattr(self, "lg", None),
            "client_rho_states": {
                int(c.id): float(getattr(c, "mu_current", getattr(c, "mu", 0.0)))
                for c in self.clients
            },
            "config": {
                "controller_mode": getattr(self.args, "controller_mode", "full"),
                "mu_base": getattr(self.args, "mu_base", None),
                "mu_min": getattr(self.args, "mu_min", None),
                "mu_max": getattr(self.args, "mu_max", None),
                "alpha_gain": getattr(self.args, "alpha_gain", None),
                "gap_tau": getattr(self.args, "gap_tau", None),
                "mu_smooth_gamma": getattr(self.args, "mu_smooth_gamma", None),
                "ema_beta": getattr(self.args, "ema_beta", None),
                "warmup_rounds": getattr(self.args, "warmup_rounds", None),
                "global_rounds": self.global_rounds,
                "local_epochs": self.local_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
            },
        }
        if any(hasattr(c, "model_per") for c in self.clients):
            ckpt["personalized_model_state_dicts"] = {
                int(c.id): c.model_per.state_dict()
                for c in self.clients if hasattr(c, "model_per")
            }
        ckpt_path = os.path.join(path, "checkpoint_final.pt")
        torch.save(ckpt, ckpt_path)
        self.log_event({"type": "checkpoint_saved", "path": ckpt_path,
                        "has_personalized": "personalized_model_state_dicts" in ckpt})

    def log_event(self, event):
        event = dict(event)
        event.setdefault("algorithm", self.algorithm)
        event.setdefault("dataset", self.dataset)
        event.setdefault("run", self.times)
        event.setdefault("seed", getattr(self.args, "current_seed", getattr(self.args, "seed", None)))
        with open(self.event_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

            # items.append((client_model, origin_grad, target_inputs))

        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args,
                            id=i,
                            train_samples=len(train_data),
                            test_samples=len(test_data),
                            train_slow=False,
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
