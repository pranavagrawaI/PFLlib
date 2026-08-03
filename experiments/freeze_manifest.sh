#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/experiments/frozen"
STATIC_SELECTION="${STATIC_SELECTION:-$ROOT/experiments/phase1_selection.csv}"
ADAPTIVE_CONTROLLER_ARGS="${ADAPTIVE_CONTROLLER_ARGS:-}"
ROUNDS="${ROUNDS:?Set ROUNDS before freezing, e.g. ROUNDS=200 bash experiments/freeze_manifest.sh}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:?Set LOCAL_EPOCHS before freezing}"
LR="${LR:?Set LR before freezing}"
PARTICIPATION_RATE="${PARTICIPATION_RATE:-1.0}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
if [[ ! -f "$STATIC_SELECTION" ]]; then
  echo "Missing static selection CSV: $STATIC_SELECTION" >&2
  exit 2
fi
python3 - "$ROOT" "$STATIC_SELECTION" "$ADAPTIVE_CONTROLLER_ARGS" "$ROUNDS" "$LOCAL_EPOCHS" "$LR" "$PARTICIPATION_RATE" "$NUM_CLIENTS" <<'PY'
import csv
import json
import shlex
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
static_selection = Path(sys.argv[2])
adaptive_args = shlex.split(sys.argv[3])
rounds = int(sys.argv[4])
local_epochs = int(sys.argv[5])
learning_rate = float(sys.argv[6])
participation_rate = float(sys.argv[7])
num_clients = int(sys.argv[8])

static_fedprox_mu = {}
static_ditto_lambda = {}
with static_selection.open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
for dataset in {row.get("dataset") for row in rows}:
    for algorithm, target, prefix in [
        ("FedProx", static_fedprox_mu, "mu_"),
        ("Ditto", static_ditto_lambda, "lam_"),
    ]:
        candidates = [row for row in rows if row.get("dataset") == dataset and row.get("algorithm") == algorithm]
        if not candidates:
            raise SystemExit(f"Missing static selection for {dataset}/{algorithm}")
        best = max(candidates, key=lambda row: float(row.get("score") or 0.0))
        run_id = best.get("run_id", "")
        if not run_id.startswith(prefix):
            raise SystemExit(f"Cannot parse coefficient from {dataset}/{algorithm} run_id={run_id}")
        target[dataset] = float(run_id[len(prefix):])

manifest = {
    "created_at_unix": time.time(),
    "static_selection_csv": str(static_selection),
    "adaptive_selection_csv": str(root / "experiments" / "phase2_selection.csv"),
    "adaptive_confirmation_csv": str(root / "experiments" / "phase2_confirm_selection.csv"),
    "static_fedprox_mu": static_fedprox_mu,
    "static_ditto_lambda": static_ditto_lambda,
    "adaptive_controller_args": adaptive_args,
    "model_architectures": {"Cifar100": "CNN", "AGNews": "Transformer"},
    "num_clients": num_clients,
    "participation_rate": participation_rate,
    "rounds": rounds,
    "local_epochs": local_epochs,
    "learning_rate": learning_rate,
    "ag_news": {"vocab_size": 32000, "max_len": 200},
    "metrics": ["server_metrics.csv", "client_controller.csv", "client_eval.csv", "events.jsonl", "h5 arrays"],
    "seeds": [0, 1, 2, 3, 4],
    "locked": True,
    "note": "Do not change settings after this file is finalized except for implementation bugs.",
}
out = root / "experiments" / "frozen" / "freeze_manifest.json"
out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
print(out)
PY
