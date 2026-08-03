#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
PHASE="phase3_controller_ablations"
DATASETS=(${CAMPAIGN_DATASETS:-Cifar100_practical AGNews_practical})
MODES=(full fixed_base no_log_ratio no_clipping no_ema mean_global_loss global_shared random_adaptive)
for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    for seed in 0 1 2; do
      run_experiment "$PHASE" "$dataset" AdaProxDitto "$seed" "${mode}_seed${seed}" --controller_mode "$mode"
    done
  done
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/phase3_ablation_summary.csv"
