#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

RUN_DIR = Path(sys.argv[1])
ALGORITHM = sys.argv[2]
RESULTS = RUN_DIR / "results"

COMMON_SERVER_COLUMNS = {
    "round",
    "algorithm",
    "dataset",
    "goal",
    "seed",
    "controller_mode",
    "selected_clients",
    "selected_client_ids",
    "global_accuracy",
    "personalized_accuracy",
    "global_train_loss",
    "personalized_train_loss",
    "reference_loss",
    "global_loss_ema",
    "coefficient_mean",
    "coefficient_variance",
    "mean_abs_coefficient_delta",
    "clipping_frequency",
    "time_cost",
    "unstable",
}
COMMON_CLIENT_EVAL_COLUMNS = {
    "round",
    "scope",
    "client_id",
    "test_samples",
    "test_accuracy",
    "test_auc",
    "train_samples",
    "train_loss",
}
ADAPTIVE_CONTROLLER_COLUMNS = {
    "round",
    "client_id",
    "algorithm",
    "dataset",
    "controller_mode",
    "Li",
    "Lg",
    "client_loss_gap_raw",
    "client_loss_gap_clipped",
    "gap_tau",
    "coefficient_prev",
    "coefficient_target",
    "coefficient_smoothed",
    "coefficient_bounded",
    "coefficient_final",
    "gap_clipped",
    "bound_clipped",
    "warmup_active",
    "abs_coefficient_delta",
    "acc_global",
}


def read_header(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"empty CSV: {path}")
        return set(reader.fieldnames)


def require(path):
    if not path.exists():
        raise SystemExit(f"missing required run artifact: {path}")

for path in [RUN_DIR / "terminal.log", RUN_DIR / "metadata.json", RUN_DIR / "command.sh", RUN_DIR / "models", RESULTS / "server_metrics.csv", RESULTS / "events.jsonl", RESULTS / "client_eval.csv"]:
    require(path)
command = (RUN_DIR / "command.sh").read_text(encoding="utf-8")
for flag, target in [
    ("--results_save_path", str(RUN_DIR / "results")),
    ("--model_save_path", str(RUN_DIR / "models")),
]:
    if flag not in command or target not in command:
        raise SystemExit(f"command.sh does not show isolated {flag}={target}")
if not list(RESULTS.glob("*.h5")):
    raise SystemExit("missing H5 result file")

server_missing = sorted(COMMON_SERVER_COLUMNS - read_header(RESULTS / "server_metrics.csv"))
if server_missing:
    raise SystemExit("server_metrics.csv missing common columns: " + ", ".join(server_missing))
client_eval_missing = sorted(COMMON_CLIENT_EVAL_COLUMNS - read_header(RESULTS / "client_eval.csv"))
if client_eval_missing:
    raise SystemExit("client_eval.csv missing columns: " + ", ".join(client_eval_missing))

if ALGORITHM in {"AdaProxFedProx", "AdaProxDitto"}:
    controller = RESULTS / "client_controller.csv"
    require(controller)
    needed = set(ADAPTIVE_CONTROLLER_COLUMNS)
    if ALGORITHM == "AdaProxDitto":
        needed.add("acc_personalized")
    missing = sorted(needed - read_header(controller))
    if missing:
        raise SystemExit("client_controller.csv missing columns: " + ", ".join(missing))
