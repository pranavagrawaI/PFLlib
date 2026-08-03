#!/usr/bin/env bash
# Shakespeare participation-shift experiment (second modality robustness).
# Adds a HIGHER-participation reference level (jr=0.5) to the jr=0.1 data we
# already have. Sweeps Ditto lambda at jr=0.5 to (a) find the reference-tuned
# lambda and (b) check whether the optimum SHIFTS vs jr=0.1 — the precondition
# for a shift win. Also runs AdaDitto at jr=0.5. Analysis then compares, at the
# shifted level jr=0.1: Ditto frozen at the jr=0.5 optimum vs AdaDitto(1 config).
# If the optimum doesn't move (Shakespeare lambda was flat for lambda>0), this
# will honestly show no shift win — that's a valid outcome.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
[[ -d "$ROOT/dataset/Shakespeare/train" ]] || { echo "Missing Shakespeare data" >&2; exit 2; }
PHASE="shakespeare_shift"
DATASET="Shakespeare"
JR="${SHAKE_REF_JR:-0.5}"   # reference (higher) participation level
ADAPTIVE_ARGS=(-ag 0.8 -musmooth 0.3 -eb 0.9 -gt 0.7 -mmax 3.0)

# lambda sweep at the reference level (1 seed, mirrors shakespeare_sweep)
for lam in 0 0.1 0.5 1.0; do
  JOIN_RATIO="$JR" run_experiment "$PHASE" "$DATASET" Ditto 0 "ditto_jr${JR}_lam${lam}" -lam "$lam"
done
# AdaDitto at the reference level (single config, 1 seed for the diagnostic)
JOIN_RATIO="$JR" run_experiment "$PHASE" "$DATASET" AdaProxDitto 0 "adaditto_jr${JR}_seed0" "${ADAPTIVE_ARGS[@]}"

python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/shakespeare_shift_selection.csv"
