#!/usr/bin/env python3
import csv
import statistics
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out_path = Path(sys.argv[2])


def f(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    frac = rank - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def final_client_stats(run_dir, preferred_scope):
    path = run_dir / "results" / "client_eval.csv"
    if not path.exists():
        return {"mean_client_accuracy": 0.0, "p10_client_accuracy": 0.0}
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("scope") == preferred_scope:
                rows.append(row)
    if not rows and preferred_scope == "personalized":
        with path.open(newline="", encoding="utf-8") as file:
            rows = [row for row in csv.DictReader(file) if row.get("scope") == "global"]
    if not rows:
        return {"mean_client_accuracy": 0.0, "p10_client_accuracy": 0.0}
    final_round = max(int(f(row.get("round"))) for row in rows)
    accs = [f(row.get("test_accuracy")) for row in rows if int(f(row.get("round"))) == final_round]
    return {
        "mean_client_accuracy": statistics.mean(accs) if accs else 0.0,
        "p10_client_accuracy": percentile(accs, 0.10),
    }

rows = []
for metrics in runs_root.glob("*/*/*/results/server_metrics.csv"):
    run_dir = metrics.parents[1]
    with metrics.open(newline="", encoding="utf-8") as file:
        data = list(csv.DictReader(file))
    if not data:
        continue
    final = data[-1]
    preferred_scope = "personalized" if final.get("algorithm") in {"Ditto", "AdaProxDitto"} else "global"
    client_stats = final_client_stats(run_dir, preferred_scope)
    personalized = f(final.get("personalized_accuracy"))
    global_acc = f(final.get("global_accuracy"))
    score = personalized or global_acc
    rows.append({
        "dataset": final.get("dataset", ""),
        "algorithm": final.get("algorithm", ""),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "score": score,
        "global_accuracy": global_acc,
        "personalized_accuracy": personalized,
        "mean_client_accuracy": client_stats["mean_client_accuracy"],
        "p10_client_accuracy": client_stats["p10_client_accuracy"],
        "coefficient_mean": f(final.get("coefficient_mean")),
        "coefficient_variance": f(final.get("coefficient_variance")),
        "mean_abs_coefficient_delta": f(final.get("mean_abs_coefficient_delta")),
        "clipping_frequency": f(final.get("clipping_frequency")),
        "unstable": int(f(final.get("unstable"))),
    })
rows.sort(key=lambda row: (row["dataset"], row["algorithm"], -row["score"]))
out_path.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "dataset",
    "algorithm",
    "run_id",
    "score",
    "global_accuracy",
    "personalized_accuracy",
    "mean_client_accuracy",
    "p10_client_accuracy",
    "coefficient_mean",
    "coefficient_variance",
    "mean_abs_coefficient_delta",
    "clipping_frequency",
    "unstable",
    "run_dir",
]
with out_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(out_path)
