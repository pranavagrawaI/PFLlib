#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
PHASE="phase2_confirm_selected"
DATASETS=(${CAMPAIGN_DATASETS:-Cifar100_practical AGNews_practical})
CONFIG_FILE="${CONFIG_FILE:-$ROOT/experiments/phase2_confirm_configs.tsv}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config file: $CONFIG_FILE" >&2
  echo "Create tab-separated rows: name<TAB>args" >&2
  exit 2
fi

for dataset in "${DATASETS[@]}"; do
  while IFS=$'\t' read -r name args; do
    [[ -z "${name:-}" || "$name" == \#* ]] && continue
    for seed in 0 1 2; do
      # shellcheck disable=SC2086
      run_experiment "$PHASE" "$dataset" AdaProxDitto "$seed" "${name}_seed${seed}" ${args:-}
    done
  done < "$CONFIG_FILE"
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/phase2_confirm_selection.csv"
