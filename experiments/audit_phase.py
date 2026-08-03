#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
RHOS = ["0", "0.001", "0.005", "0.01", "0.03", "0.05", "0.08", "0.1", "0.2", "0.5", "1.0", "2.0", "3.0"]
DATASETS4 = ["Cifar100_practical", "Cifar100_pathological", "AGNews_practical", "AGNews_pathological"]
DATASETS2 = ["Cifar100_practical", "AGNews_practical"]
METHODS4 = ["FedProx", "AdaProxFedProx", "Ditto", "AdaProxDitto"]
MODES = ["full", "fixed_base", "no_log_ratio", "no_clipping", "no_ema", "mean_global_loss", "global_shared", "random_adaptive"]
ADAPTIVE = {"AdaProxFedProx", "AdaProxDitto"}

SERVER_COLUMNS = {
    "round", "algorithm", "dataset", "goal", "seed", "controller_mode", "selected_clients",
    "selected_client_ids", "global_accuracy", "personalized_accuracy", "global_train_loss",
    "personalized_train_loss", "reference_loss", "global_loss_ema", "coefficient_mean",
    "coefficient_variance", "mean_abs_coefficient_delta", "clipping_frequency", "time_cost",
    "unstable",
}
CLIENT_EVAL_COLUMNS = {
    "round", "scope", "client_id", "test_samples", "test_accuracy", "test_auc",
    "train_samples", "train_loss",
}
CONTROLLER_COLUMNS = {
    "round", "client_id", "algorithm", "dataset", "controller_mode", "Li", "Lg",
    "client_loss_gap_raw", "client_loss_gap_clipped", "gap_tau", "coefficient_prev",
    "coefficient_target", "coefficient_smoothed", "coefficient_bounded", "coefficient_final",
    "gap_clipped", "bound_clipped", "warmup_active", "abs_coefficient_delta", "acc_global",
}
SUMMARY_COLUMNS = {
    "dataset", "algorithm", "run_id", "score", "global_accuracy", "personalized_accuracy",
    "mean_client_accuracy", "p10_client_accuracy", "coefficient_mean", "coefficient_variance",
    "mean_abs_coefficient_delta", "clipping_frequency", "unstable", "run_dir",
}


def fail(message):
    raise SystemExit(f"audit failed: {message}")


def rows(path):
    if not path.exists():
        fail(f"missing {path}")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            fail(f"empty CSV header: {path}")
        data = list(reader)
    if not data:
        fail(f"no rows in {path}")
    return reader.fieldnames, data


def require(path):
    if not path.exists():
        fail(f"missing {path}")
    return path


def run_dir(phase, dataset, algorithm, run_id):
    return RUNS / phase / dataset / algorithm / run_id


def audit_run(phase, dataset, algorithm, run_id):
    d = run_dir(phase, dataset, algorithm, run_id)
    require(d / "terminal.log")
    require(d / "metadata.json")
    require(d / "dataset_config.json")
    require(d / "command.sh")
    require(d / "git_status.txt")
    require(d / "models")
    command = (d / "command.sh").read_text(encoding="utf-8")
    for flag, target in [
        ("--results_save_path", str(d / "results")),
        ("--model_save_path", str(d / "models")),
    ]:
        if flag not in command or target not in command:
            fail(f"{d}: command.sh does not show isolated {flag}={target}")
    if not list((d / "results").glob("*.h5")):
        fail(f"missing H5 result under {d / 'results'}")
    server_header, server_rows = rows(d / "results" / "server_metrics.csv")
    missing = sorted(SERVER_COLUMNS - set(server_header))
    if missing:
        fail(f"{d}: server_metrics.csv missing {missing}")
    eval_header, _ = rows(d / "results" / "client_eval.csv")
    missing = sorted(CLIENT_EVAL_COLUMNS - set(eval_header))
    if missing:
        fail(f"{d}: client_eval.csv missing {missing}")
    if not (d / "results" / "events.jsonl").read_text(encoding="utf-8").strip():
        fail(f"{d}: events.jsonl is empty")
    if algorithm in ADAPTIVE:
        header, data = rows(d / "results" / "client_controller.csv")
        needed = set(CONTROLLER_COLUMNS)
        if algorithm == "AdaProxDitto":
            needed.add("acc_personalized")
        missing = sorted(needed - set(header))
        if missing:
            fail(f"{d}: client_controller.csv missing {missing}")
        if not any(r.get("client_loss_gap_raw", "") != "" for r in data):
            fail(f"{d}: client_controller.csv lacks loss-gap values")
    return d, set(server_header), server_rows


def audit_summary(path, min_rows):
    header, data = rows(path)
    missing = sorted(SUMMARY_COLUMNS - set(header))
    if missing:
        fail(f"{path}: summary missing {missing}")
    if len(data) < min_rows:
        fail(f"{path}: expected at least {min_rows} rows, found {len(data)}")


def audit_phase0():
    schemas = []
    for dataset in DATASETS4:
        for algorithm in METHODS4:
            _, schema, server_rows = audit_run("phase0_smoke", dataset, algorithm, "seed0")
            schemas.append(schema)
            if algorithm in ADAPTIVE:
                final = server_rows[-1]
                for col in ["global_loss_ema", "coefficient_variance", "mean_abs_coefficient_delta", "clipping_frequency"]:
                    if final.get(col, "") == "":
                        fail(f"phase0 {dataset}/{algorithm}: empty {col}")
    _, _, partial_rows = audit_run("phase0_smoke", "Cifar100_practical_100", "AdaProxDitto", "partial_100_10")
    if not any(int(float(r.get("selected_clients") or 0)) == 10 for r in partial_rows):
        fail("phase0 partial participation did not log selected_clients=10")
    if any(s != schemas[0] for s in schemas[1:]):
        fail("phase0 server metric schemas differ across algorithms")


def audit_phase1():
    count = 0
    for dataset in DATASETS4:
        for rho in RHOS:
            audit_run("phase1_static_sweeps", dataset, "FedProx", f"mu_{rho}")
            audit_run("phase1_static_sweeps", dataset, "Ditto", f"lam_{rho}")
            count += 2
    if count != 104:
        fail(f"phase1 expected 104 runs, counted {count}")
    audit_summary(ROOT / "experiments" / "phase1_selection.csv", 104)


def audit_phase2_scan():
    scan_ids = (
        ["default"]
        + [f"alpha_{v}" for v in ["0.1", "0.3", "0.5", "0.8", "1.2", "1.6"]]
        + [f"gamma_{v}" for v in ["0.05", "0.1", "0.3", "0.5", "1.0"]]
        + [f"beta_{v}" for v in ["0.0", "0.5", "0.8", "0.9", "0.99"]]
        + [f"tau_{v}" for v in ["0.1", "0.3", "0.5", "0.7", "1.0", "2.0"]]
        + [f"rhomax_{v}" for v in ["0.2", "0.5", "1.0", "3.0", "10.0"]]
    )
    for dataset in DATASETS2:
        for run_id in scan_ids:
            audit_run("phase2_adaptive_scan", dataset, "AdaProxDitto", run_id)
    audit_summary(ROOT / "experiments" / "phase2_selection.csv", len(DATASETS2) * len(scan_ids))


def confirm_names():
    path = ROOT / "experiments" / "phase2_confirm_configs.tsv"
    if not path.exists():
        fail(f"missing {path}")
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        names.append(raw.split("\t", 1)[0])
    if not names:
        fail(f"{path}: no confirmation configs")
    return names


def audit_phase2_confirm():
    names = confirm_names()
    for dataset in DATASETS2:
        for name in names:
            for seed in range(3):
                audit_run("phase2_confirm_selected", dataset, "AdaProxDitto", f"{name}_seed{seed}")
    audit_summary(ROOT / "experiments" / "phase2_confirm_selection.csv", len(DATASETS2) * len(names) * 3)


def audit_phase3():
    for dataset in DATASETS2:
        for mode in MODES:
            for seed in range(3):
                audit_run("phase3_controller_ablations", dataset, "AdaProxDitto", f"{mode}_seed{seed}")
    audit_summary(ROOT / "experiments" / "phase3_ablation_summary.csv", len(DATASETS2) * len(MODES) * 3)


def audit_freeze():
    path = ROOT / "experiments" / "frozen" / "freeze_manifest.json"
    data = json.loads(require(path).read_text(encoding="utf-8"))
    if data.get("locked") is not True:
        fail("freeze manifest is not locked")
    for key in ["static_fedprox_mu", "static_ditto_lambda"]:
        got = data.get(key, {})
        missing = sorted(set(DATASETS4) - set(got))
        if missing:
            fail(f"freeze manifest {key} missing {missing}")
    for key in ["adaptive_controller_args", "model_architectures", "num_clients", "participation_rate", "rounds", "local_epochs", "learning_rate", "ag_news", "metrics", "seeds"]:
        if key not in data:
            fail(f"freeze manifest missing {key}")


def audit_core():
    ids = {
        "FedProx": "fedprox_seed",
        "AdaProxFedProx": "adaprox_seed",
        "Ditto": "ditto_seed",
        "AdaProxDitto": "adaditto_seed",
    }
    for dataset in DATASETS4:
        for algorithm, prefix in ids.items():
            for seed in range(5):
                audit_run("core_5seed", dataset, algorithm, f"{prefix}{seed}")
    audit_summary(ROOT / "experiments" / "core_5seed_summary.csv", len(DATASETS4) * len(ids) * 5)


AUDITS = {
    "phase0": audit_phase0,
    "phase1": audit_phase1,
    "phase2_scan": audit_phase2_scan,
    "phase2_confirm": audit_phase2_confirm,
    "phase3": audit_phase3,
    "freeze": audit_freeze,
    "core": audit_core,
}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    selected = list(AUDITS) if target == "all" else [target]
    for name in selected:
        if name not in AUDITS:
            fail(f"unknown audit target {name}; expected one of {', '.join(AUDITS)} or all")
        AUDITS[name]()
        print(f"ok {name}")


if __name__ == "__main__":
    main()
