#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ONE="$ROOT/experiments/run_one.sh"
DEVICE_ID="${DEVICE_ID:-0}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
# ROUNDS / LR / BATCH_SIZE default per-dataset (see dataset_rounds/dataset_lr/
# dataset_batch); leave empty unless the caller overrides them explicitly.
ROUNDS="${ROUNDS:-}"
LR="${LR:-}"
EVAL_GAP="${EVAL_GAP:-1}"
BATCH_SIZE="${BATCH_SIZE:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

dataset_model() {
  case "$1" in
    AGNews*) echo "Transformer" ;;
    Shakespeare*) echo "LSTM" ;;
    *) echo "CNN" ;;
  esac
}

dataset_classes() {
  case "$1" in
    AGNews*) echo "4" ;;
    Cifar100*) echo "100" ;;
    Shakespeare*) echo "80" ;;  # next-character prediction, 80-symbol vocab
    *) echo "10" ;;
  esac
}

# Natural client count for datasets with a fixed partition (Shakespeare = one
# client per LEAF speaker). CIFAR/AGNews use 20 unless NUM_CLIENTS overrides.
dataset_clients() {
  case "$1" in
    Shakespeare*) echo "118" ;;
    *) echo "20" ;;
  esac
}

# Per-dataset frozen learning rate / batch size. Source of truth for these two
# settings; an explicit LR=/BATCH_SIZE= in the environment overrides them.
dataset_lr() {
  case "$1" in
    AGNews*) echo "0.005" ;;
    Shakespeare*) echo "0.5" ;;  # LEAF-style high LR for char-LSTM (calibrated)
    *) echo "0.01" ;;
  esac
}

dataset_batch() {
  case "$1" in
    AGNews*) echo "128" ;;
    *) echo "64" ;;
  esac
}

# AG News (Transformer) is ~35x slower per round than CIFAR (CNN), so it runs
# fewer rounds and a shorter sequence length. CIFAR keeps the full budget.
dataset_rounds() {
  case "$1" in
    AGNews*) echo "50" ;;
    Shakespeare*) echo "40" ;;  # char-LSTM converges fast; eval over 118 clients is costly
    *) echo "200" ;;
  esac
}

dataset_maxlen() {
  case "$1" in
    AGNews*) echo "64" ;;
    Shakespeare*) echo "80" ;;  # 80-character input sequences
    *) echo "200" ;;
  esac
}

ensure_splits() {
  if [[ "${SKIP_DATA_PREP:-0}" == "1" ]]; then
    return 0
  fi
  cd "$ROOT/dataset"
  "$PYTHON_BIN" generate_Cifar100.py --out_dir Cifar100_practical/ --num_clients 20 --partition dir --balance false --class_per_client 10
  "$PYTHON_BIN" generate_Cifar100.py --out_dir Cifar100_pathological/ --num_clients 20 --partition pat --balance false --class_per_client 10
  "$PYTHON_BIN" generate_Cifar100.py --out_dir Cifar100_practical_100/ --num_clients 100 --partition dir --balance false --class_per_client 10
  "$PYTHON_BIN" generate_AGNews.py --out_dir AGNews_practical/ --num_clients 20 --partition dir --balance true --max_len 64
  "$PYTHON_BIN" generate_AGNews.py --out_dir AGNews_pathological/ --num_clients 20 --partition pat --balance true --class_per_client 1 --max_len 64
}

run_experiment() {
  local phase="$1" dataset="$2" algorithm="$3" seed="$4" run_id="$5"
  shift 5
  local data_args=(-ncl "$(dataset_classes "$dataset")")
  if [[ "$dataset" == AGNews* ]]; then
    data_args+=(-vs 32000 -ml "$(dataset_maxlen "$dataset")")
  elif [[ "$dataset" == Shakespeare* ]]; then
    data_args+=(-vs 80 -ml "$(dataset_maxlen "$dataset")" -fd 256)
  fi
  local lr="${LR:-$(dataset_lr "$dataset")}"
  local batch="${BATCH_SIZE:-$(dataset_batch "$dataset")}"
  local rounds="${ROUNDS:-$(dataset_rounds "$dataset")}"
  PHASE="$phase" DATASET="$dataset" ALGORITHM="$algorithm" RUN_ID="$run_id" \
  MODEL="$(dataset_model "$dataset")" NUM_CLIENTS="${NUM_CLIENTS:-$(dataset_clients "$dataset")}" JOIN_RATIO="${JOIN_RATIO:-1.0}" \
  ROUNDS="$rounds" LOCAL_EPOCHS="$LOCAL_EPOCHS" LR="$lr" EVAL_GAP="$EVAL_GAP" \
  DEVICE_ID="$DEVICE_ID" SEED="$seed" BATCH_SIZE="$batch" PYTHON_BIN="$PYTHON_BIN" \
  SKIP_EXISTING="${SKIP_EXISTING:-1}" FORCE="${FORCE:-0}" \
  bash "$RUN_ONE" "${data_args[@]}" "$@"
}
