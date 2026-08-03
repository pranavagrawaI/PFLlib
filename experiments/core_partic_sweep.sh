#!/usr/bin/env bash
# Per-condition mu/lambda re-tuning for the partial-participation core table.
# The Phase-1 sweep tuned FedProx mu / Ditto lambda at FULL participation; the
# optimum shifts with participation (FedProx mu tends to rise as fewer clients
# update per round => more drift to regularize). So we re-sweep at each jr level
# and feed the per-condition argmax into core_participation as the *fairly
# tuned* fixed baselines. Coarse 6-point grid (0..1.0), 1 seed, like Phase 1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
PHASE="core_partic_sweep"
DATASETS=(${CAMPAIGN_DATASETS:-Cifar100_practical Cifar100_pathological})
JR_LEVELS=(${CORE_JR_LEVELS:-0.5 0.2})
RHOS=(0 0.001 0.01 0.1 0.5 1.0)
for dataset in "${DATASETS[@]}"; do
  for jr in "${JR_LEVELS[@]}"; do
    for rho in "${RHOS[@]}"; do
      JOIN_RATIO="$jr" run_experiment "$PHASE" "$dataset" FedProx 0 "fedprox_jr${jr}_mu${rho}"  -mu  "$rho"
      JOIN_RATIO="$jr" run_experiment "$PHASE" "$dataset" Ditto   0 "ditto_jr${jr}_lam${rho}"   -lam "$rho"
    done
  done
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/core_partic_sweep_selection.csv"
