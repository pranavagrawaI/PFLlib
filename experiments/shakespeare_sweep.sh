#!/usr/bin/env bash
# Shakespeare (text / char-LSTM) Ditto lambda sweep — second modality.
# Establishes (a) that the personalization coefficient lambda matters on text,
# and (b) the best fixed lambda baseline for the AdaDitto comparison. Natural
# per-speaker non-IID (118 LEAF clients), partial participation jr=0.1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
[[ -d "$ROOT/dataset/Shakespeare/train" ]] || { echo "Missing Shakespeare data" >&2; exit 2; }
PHASE="shakespeare_sweep"
DATASET="Shakespeare"
JR="${SHAKE_JR:-0.1}"
for lam in 0 0.1 0.5 1.0; do
  JOIN_RATIO="$JR" run_experiment "$PHASE" "$DATASET" Ditto 0 "ditto_jr${JR}_lam${lam}" -lam "$lam"
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/shakespeare_sweep_selection.csv"
