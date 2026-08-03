#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
PHASE="phase2_adaptive_scan"
DATASETS=(${CAMPAIGN_DATASETS:-Cifar100_practical AGNews_practical})
for dataset in "${DATASETS[@]}"; do
  run_experiment "$PHASE" "$dataset" AdaProxDitto 0 "default"
  for v in 0.1 0.3 0.5 0.8 1.2 1.6; do run_experiment "$PHASE" "$dataset" AdaProxDitto 0 "alpha_${v}" -ag "$v"; done
  for v in 0.05 0.1 0.3 0.5 1.0; do run_experiment "$PHASE" "$dataset" AdaProxDitto 0 "gamma_${v}" -musmooth "$v"; done
  for v in 0.0 0.5 0.8 0.9 0.99; do run_experiment "$PHASE" "$dataset" AdaProxDitto 0 "beta_${v}" -eb "$v"; done
  for v in 0.1 0.3 0.5 0.7 1.0 2.0; do run_experiment "$PHASE" "$dataset" AdaProxDitto 0 "tau_${v}" -gt "$v"; done
  for v in 0.2 0.5 1.0 3.0 10.0; do run_experiment "$PHASE" "$dataset" AdaProxDitto 0 "rhomax_${v}" -mmax "$v"; done
done
python3 "$ROOT/experiments/summarize_phase.py" "$ROOT/experiments/runs/$PHASE" "$ROOT/experiments/phase2_selection.csv"
cat <<'MSG'
Phase 2 wide scan complete. Review experiments/phase2_selection.csv, choose the fixed controller and 1-2 nearby alternatives, then run:
  bash experiments/phase2_confirm_selected.sh
Edit experiments/phase2_confirm_configs.tsv with rows like:
  selected<TAB>-ag 0.8 -musmooth 0.3
  nearby_alpha<TAB>-ag 0.5
Then run:
  bash experiments/phase2_confirm_selected.sh
MSG
