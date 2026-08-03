#!/usr/bin/env bash
# Master driver for the experiment campaign (single-GPU, serial).
#
#   bash experiments/run_campaign.sh <target>
#
# targets: phase0 | timing | phase1 | phase2 | phase2_confirm | phase3 | freeze | core
#
# Frozen real settings (override by exporting before the call):
#   ROUNDS=200 EVAL_GAP=5 LOCAL_EPOCHS=1   (phase0 forces ROUNDS=5 EVAL_GAP=1)
# Per-dataset LR/batch come from common.sh (dataset_lr/dataset_batch).
# Everything is resumable: re-launching a target skips completed runs and redoes
# incomplete ones (SKIP_EXISTING=1). Recommended for long phases:
#   tmux new -s campaign 'bash experiments/run_campaign.sh phase1'
#   # or:  nohup bash experiments/run_campaign.sh phase1 &
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET="${1:?usage: run_campaign.sh <phase0|timing|phase1|phase2|phase2_confirm|phase3|freeze|core>}"

export DEVICE_ID="${DEVICE_ID:-0}"
export PYTHON_BIN="${PYTHON_BIN:-/home/hipra/miniconda3/envs/pfllib/bin/python}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
# ROUNDS intentionally unset: per-dataset defaults apply (CIFAR 200, AG News 50;
# see common.sh dataset_rounds). phase0 overrides to ROUNDS=5 below.
export EVAL_GAP="${EVAL_GAP:-5}"
export LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"

mkdir -p "$ROOT/experiments/logs"
LOG="$ROOT/experiments/logs/${TARGET}_$(date +%Y%m%d_%H%M%S).log"

phase_progress() {
  local phase="$1"
  local base="$ROOT/experiments/runs/$phase"
  [[ -d "$base" ]] || { echo "  (no runs yet under $phase)"; return; }
  local total completed
  total=$(find "$base" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | wc -l)
  completed=$(grep -rl '"status": "completed"' "$base"/*/*/*/status.json 2>/dev/null | wc -l || true)
  echo "  progress[$phase]: ${completed}/${total} completed"
}

run_target() {
  case "$TARGET" in
    phase0)
      ROUNDS=5 EVAL_GAP=1 bash "$ROOT/experiments/phase0_smoke.sh"
      phase_progress phase0_smoke
      ;;
    timing)
      # 4-run probe at the real round budget to measure per-run cost.
      source "$ROOT/experiments/common.sh"
      ensure_splits
      for ds in Cifar100_practical AGNews_practical; do
        run_experiment timing "$ds" FedProx 0 "fedprox" -mu 0.05
        run_experiment timing "$ds" AdaProxDitto 0 "adaditto"
      done
      phase_progress timing
      "$PYTHON_BIN" "$ROOT/experiments/estimate_time.py" || true
      ;;
    phase1)
      bash "$ROOT/experiments/phase1_static_sweeps.sh"
      phase_progress phase1_static_sweeps
      ;;
    phase2)
      bash "$ROOT/experiments/phase2_adaptive_scan.sh"
      phase_progress phase2_adaptive_scan
      ;;
    phase2_confirm)
      bash "$ROOT/experiments/phase2_confirm_selected.sh"
      phase_progress phase2_confirm_selected
      ;;
    phase3)
      bash "$ROOT/experiments/phase3_controller_ablations.sh"
      phase_progress phase3_controller_ablations
      ;;
    freeze)
      : "${ADAPTIVE_CONTROLLER_ARGS:?set ADAPTIVE_CONTROLLER_ARGS, e.g. ADAPTIVE_CONTROLLER_ARGS=\"-ag 0.8 -musmooth 0.3\"}"
      ROUNDS="$ROUNDS" LOCAL_EPOCHS="$LOCAL_EPOCHS" LR="${LR:-0.01}" \
        ADAPTIVE_CONTROLLER_ARGS="$ADAPTIVE_CONTROLLER_ARGS" \
        bash "$ROOT/experiments/freeze_manifest.sh"
      ;;
    core)
      local manifest="$ROOT/experiments/frozen/freeze_manifest.json"
      [[ -f "$manifest" ]] || { echo "Missing $manifest — run 'freeze' first." >&2; exit 2; }
      bash "$ROOT/experiments/core_5seed.sh" "$manifest"
      phase_progress core_5seed
      ;;
    *)
      echo "unknown target: $TARGET" >&2
      exit 2
      ;;
  esac
}

echo "=== campaign target=$TARGET  rounds=per-dataset(CIFAR 200/AGNews 50) EVAL_GAP=$EVAL_GAP  log=$LOG ==="
run_target 2>&1 | tee "$LOG"
echo "=== done: $TARGET (log: $LOG) ==="
