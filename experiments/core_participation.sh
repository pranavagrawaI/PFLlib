#!/usr/bin/env bash
# Core comparison table under partial participation, with FAIRLY-tuned baselines.
# FedProx mu / Ditto lambda are taken from tuned_baselines.json — the per-(split,
# jr) argmax of core_partic_sweep (NOT the full-participation Phase-1 optima,
# which don't transfer: Ditto lambda jumps 0.0->1.0 at jr=0.2 pathological).
# The adaptive controller (AdaProx*) uses a SINGLE default config across every
# condition (manifest adaptive_controller_args) — its whole value proposition is
# "no per-condition tuning". 4 algos x 5 seeds x 2 splits x jr{0.5,0.2}=80 runs.
set -euo pipefail
MANIFEST="${1:?usage: bash experiments/core_participation.sh /abs/path/freeze_manifest.json}"
# Resolve to an absolute path BEFORE sourcing common.sh (ensure_splits cd's away,
# which would break a relative path).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNED="${TUNED_BASELINES:-$_SCRIPT_DIR/tuned_baselines.json}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
PHASE="core_participation"
[[ -f "$TUNED" ]] || { echo "Missing tuned baselines: $TUNED" >&2; exit 2; }

manifest_value() {
  python3 - "$MANIFEST" "$1" "${2:-}" <<'PY'
import json, sys
path, key, dataset = sys.argv[1:]
v = json.load(open(path, encoding="utf-8"))[key]
if dataset: v = v[dataset]
print("\n".join(map(str, v)) if isinstance(v, list) else v)
PY
}
tuned_value() {  # algo dataset jr  -> per-condition mu/lambda
  python3 - "$TUNED" "$1" "$2" "$3" <<'PY'
import json, sys
path, algo, dataset, jr = sys.argv[1:]
print(json.load(open(path, encoding="utf-8"))[algo][dataset][jr])
PY
}

mapfile -t ADAPTIVE_ARGS < <(manifest_value adaptive_controller_args)
DATASETS=(${CAMPAIGN_DATASETS:-Cifar100_practical Cifar100_pathological})
JR_LEVELS=(${CORE_JR_LEVELS:-0.5 0.2})
for dataset in "${DATASETS[@]}"; do
  for jr in "${JR_LEVELS[@]}"; do
    fedprox_mu="$(tuned_value FedProx "$dataset" "$jr")"
    ditto_lam="$(tuned_value Ditto "$dataset" "$jr")"
    for seed in 0 1 2 3 4; do
      JOIN_RATIO="$jr" run_experiment "$PHASE" "$dataset" FedProx        "$seed" "fedprox_jr${jr}_seed${seed}"  -mu  "$fedprox_mu"
      JOIN_RATIO="$jr" run_experiment "$PHASE" "$dataset" AdaProxFedProx "$seed" "adaprox_jr${jr}_seed${seed}"  "${ADAPTIVE_ARGS[@]}"
      JOIN_RATIO="$jr" run_experiment "$PHASE" "$dataset" Ditto          "$seed" "ditto_jr${jr}_seed${seed}"    -lam "$ditto_lam"
      JOIN_RATIO="$jr" run_experiment "$PHASE" "$dataset" AdaProxDitto   "$seed" "adaditto_jr${jr}_seed${seed}" "${ADAPTIVE_ARGS[@]}"
    done
  done
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/core_participation_summary.csv"
