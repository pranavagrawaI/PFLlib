#!/usr/bin/env bash
# Shakespeare core comparison (second modality): AdaDitto (single default config)
# vs Ditto at its best-swept lambda, on text/char-LSTM personalization.
# Reads the best Ditto lambda from shakespeare_sweep_selection.csv. Reports
# personalized next-character accuracy, 3 seeds, partial participation jr=0.1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
PHASE="shakespeare_core"
DATASET="Shakespeare"
JR="${SHAKE_JR:-0.1}"
SWEEP_CSV="${SHAKE_SWEEP_CSV:-$ROOT/experiments/shakespeare_sweep_selection.csv}"

# best Ditto lambda = argmax score over the sweep (override with DITTO_LAM=)
ditto_lam="${DITTO_LAM:-}"
if [[ -z "$ditto_lam" ]]; then
  ditto_lam="$(python3 - "$SWEEP_CSV" <<'PY'
import csv, re, sys
best=(None, -1)
for r in csv.DictReader(open(sys.argv[1])):
    m=re.search(r'lam([0-9.]+)', r.get('run_id',''))
    try: s=float(r.get('score') or -1)
    except: s=-1
    if m and s>best[1]: best=(m.group(1), s)
print(best[0] if best[0] is not None else "0.1")
PY
)"
fi
echo "[shakespeare_core] using Ditto lambda=$ditto_lam (best from sweep)"

ADAPTIVE_ARGS=(-ag 0.8 -musmooth 0.3 -eb 0.9 -gt 0.7 -mmax 3.0)
for seed in 0 1 2; do
  JOIN_RATIO="$JR" run_experiment "$PHASE" "$DATASET" Ditto        "$seed" "ditto_jr${JR}_seed${seed}"    -lam "$ditto_lam"
  JOIN_RATIO="$JR" run_experiment "$PHASE" "$DATASET" AdaProxDitto "$seed" "adaditto_jr${JR}_seed${seed}" "${ADAPTIVE_ARGS[@]}"
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/shakespeare_core_summary.csv"
