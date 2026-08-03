#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
PHASE="phase1_static_sweeps"
DATASETS=(${CAMPAIGN_DATASETS:-Cifar100_practical Cifar100_pathological AGNews_practical AGNews_pathological})
RHOS=(0 0.001 0.005 0.01 0.03 0.05 0.08 0.1 0.2 0.5 1.0 2.0 3.0)
for dataset in "${DATASETS[@]}"; do
  for rho in "${RHOS[@]}"; do
    run_experiment "$PHASE" "$dataset" FedProx 0 "mu_${rho}" -mu "$rho"
    run_experiment "$PHASE" "$dataset" Ditto 0 "lam_${rho}" -lam "$rho"
  done
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/phase1_selection.csv"
