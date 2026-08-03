#!/usr/bin/env bash
# Dropout / partial-participation robustness sweep.
#
# Tests the paper's robustness claim on the FedProx/Ditto home turf: a proximal
# coefficient tuned ONCE at full participation (cdr=0, jr=1.0) goes stale when
# clients drop or only partially participate, whereas the adaptive controller
# tracks the moving optimum. So the fixed baseline is frozen at its Phase-1
# cdr=0 optimum and we watch whether it degrades vs the controller as we stress
# two axes:
#   - client_drop_rate (cdr): selected clients that drop out (do no work) per round
#   - join_ratio (jr): fraction of clients sampled per round (partial participation)
#
# Contrast: AdaProxDitto (controller, default frozen args) vs Ditto (best fixed
# lambda from phase1_selection.csv), both CIFAR splits, 3 seeds.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
PHASE="phase_dropout"
DATASETS=(${CAMPAIGN_DATASETS:-Cifar100_practical Cifar100_pathological})

# Frozen Ditto lambda = phase-1 argmax per split (the "tuned once" baseline).
ditto_lambda() {
  case "$1" in
    Cifar100_practical)    echo 0.5 ;;
    Cifar100_pathological) echo 0.0 ;;
    *) echo 0.1 ;;  # fallback
  esac
}

# Light cross over both axes, centred on the full-participation baseline:
#   cdr arm (jr=1.0):  0.0 -> 0.3 -> 0.6
#   jr  arm (cdr=0.0): 1.0 -> 0.5 -> 0.2
#   one crossed corner: cdr=0.3, jr=0.5
# Each entry is "cdr:jr".
CONDITIONS=(0.0:1.0 0.3:1.0 0.6:1.0 0.0:0.5 0.0:0.2 0.3:0.5)

for dataset in "${DATASETS[@]}"; do
  lam="$(ditto_lambda "$dataset")"
  for cond in "${CONDITIONS[@]}"; do
    cdr="${cond%%:*}"; jr="${cond##*:}"
    tag="cdr${cdr}_jr${jr}"
    for seed in 0 1 2; do
      # Ditto: frozen best lambda; controller is OFF (pure fixed baseline).
      JOIN_RATIO="$jr" run_experiment "$PHASE" "$dataset" Ditto "$seed" \
        "ditto_${tag}_seed${seed}" -lam "$lam" -cdr "$cdr"
      # AdaProxDitto: adaptive controller with default frozen args.
      JOIN_RATIO="$jr" run_experiment "$PHASE" "$dataset" AdaProxDitto "$seed" \
        "ada_${tag}_seed${seed}" -cdr "$cdr"
    done
  done
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/phase_dropout_summary.csv"
