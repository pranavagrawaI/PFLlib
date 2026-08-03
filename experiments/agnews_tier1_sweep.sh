#!/usr/bin/env bash
# AGNews Tier-1 static sweep (coarse) on the practical split.
# Doubles as (a) the fixed-mu/lambda baselines for the AGNews core table and
# (b) the brittleness-on-text evidence. Coarse 6-point grid (vs CIFAR's 13)
# to keep the 14x-per-run cost bounded; 1 seed, as in the CIFAR Phase-1 sweep.
# Run with PFLLIB_AMP=bf16 PFLLIB_FORCE_MATH_SDPA=1 (set by the launcher).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
PHASE="agnews_tier1_sweep"
DATASET="${AGNEWS_SWEEP_DATASET:-AGNews_practical}"
RHOS=(0 0.01 0.05 0.1 0.5 1.0)
for rho in "${RHOS[@]}"; do
  run_experiment "$PHASE" "$DATASET" FedProx 0 "fedprox_mu_${rho}" -mu "$rho"
  run_experiment "$PHASE" "$DATASET" Ditto   0 "ditto_lam_${rho}"  -lam "$rho"
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/agnews_tier1_sweep_selection.csv"
