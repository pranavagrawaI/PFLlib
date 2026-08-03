#!/usr/bin/env bash
set -euo pipefail
MANIFEST="${1:?usage: bash experiments/core_5seed.sh experiments/frozen/freeze_manifest.json}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
PHASE="core_5seed"

manifest_value() {
  local key="$1" dataset="${2:-}"
  python3 - "$MANIFEST" "$key" "$dataset" <<'PY'
import json, sys
path, key, dataset = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    manifest = json.load(f)
value = manifest[key]
if dataset:
    value = value[dataset]
if isinstance(value, list):
    for item in value:
        print(item)
else:
    print(value)
PY
}

mapfile -t ADAPTIVE_ARGS < <(manifest_value adaptive_controller_args)
DATASETS=(${CAMPAIGN_DATASETS:-Cifar100_practical Cifar100_pathological AGNews_practical AGNews_pathological})
for dataset in "${DATASETS[@]}"; do
  fedprox_mu="$(manifest_value static_fedprox_mu "$dataset")"
  ditto_lam="$(manifest_value static_ditto_lambda "$dataset")"
  for seed in 0 1 2 3 4; do
    run_experiment "$PHASE" "$dataset" FedProx "$seed" "fedprox_seed${seed}" -mu "$fedprox_mu"
    run_experiment "$PHASE" "$dataset" AdaProxFedProx "$seed" "adaprox_seed${seed}" "${ADAPTIVE_ARGS[@]}"
    run_experiment "$PHASE" "$dataset" Ditto "$seed" "ditto_seed${seed}" -lam "$ditto_lam"
    run_experiment "$PHASE" "$dataset" AdaProxDitto "$seed" "adaditto_seed${seed}" "${ADAPTIVE_ARGS[@]}"
  done
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/core_5seed_summary.csv"
