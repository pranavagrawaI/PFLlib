#!/usr/bin/env bash
# Controller robustness scan: AdaProxDitto (AdaDitto) under stress controller
# configs at a chosen (dataset, participation). Tests that the default was not
# hand-picked for one condition. 11 configs = default + low/high extreme of each
# controller knob (alpha gain, smoothing gamma, EMA beta, gap tau, rho_max).
# Env: SCAN_DATASET, SCAN_JR, SCAN_SEEDS (space-separated), SCAN_PHASE.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
DATASET="${SCAN_DATASET:?set SCAN_DATASET}"
JR="${SCAN_JR:?set SCAN_JR}"
PHASE="${SCAN_PHASE:-controller_scan_${DATASET}_jr${JR}}"
read -r -a SEEDS <<< "${SCAN_SEEDS:-0 1 2}"

# name<TAB>args ; default first so it's easy to rank against the extremes
CONFIGS=(
  "default|-ag 0.8 -musmooth 0.3 -eb 0.9 -gt 0.7 -mmax 3.0"
  "ag_lo|-ag 0.1 -musmooth 0.3 -eb 0.9 -gt 0.7 -mmax 3.0"
  "ag_hi|-ag 1.6 -musmooth 0.3 -eb 0.9 -gt 0.7 -mmax 3.0"
  "gamma_lo|-ag 0.8 -musmooth 0.05 -eb 0.9 -gt 0.7 -mmax 3.0"
  "gamma_hi|-ag 0.8 -musmooth 1.0 -eb 0.9 -gt 0.7 -mmax 3.0"
  "beta_lo|-ag 0.8 -musmooth 0.3 -eb 0.0 -gt 0.7 -mmax 3.0"
  "beta_hi|-ag 0.8 -musmooth 0.3 -eb 0.99 -gt 0.7 -mmax 3.0"
  "tau_lo|-ag 0.8 -musmooth 0.3 -eb 0.9 -gt 0.1 -mmax 3.0"
  "tau_hi|-ag 0.8 -musmooth 0.3 -eb 0.9 -gt 2.0 -mmax 3.0"
  "rhomax_lo|-ag 0.8 -musmooth 0.3 -eb 0.9 -gt 0.7 -mmax 0.2"
  "rhomax_hi|-ag 0.8 -musmooth 0.3 -eb 0.9 -gt 0.7 -mmax 10.0"
)
for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"; args="${entry#*|}"
  for seed in "${SEEDS[@]}"; do
    # shellcheck disable=SC2086
    JOIN_RATIO="$JR" run_experiment "$PHASE" "$DATASET" AdaProxDitto "$seed" "${name}_seed${seed}" $args
  done
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/${PHASE}_summary.csv"
