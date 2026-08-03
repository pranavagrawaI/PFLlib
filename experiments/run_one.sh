#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${PHASE:?PHASE is required}"
DATASET="${DATASET:?DATASET is required}"
ALGORITHM="${ALGORITHM:?ALGORITHM is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
MODEL="${MODEL:-CNN}"
ROUNDS="${ROUNDS:-5}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
LR="${LR:-0.005}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
JOIN_RATIO="${JOIN_RATIO:-1.0}"
EVAL_GAP="${EVAL_GAP:-1}"
DEVICE_ID="${DEVICE_ID:-0}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GOAL="${GOAL:-$PHASE}"
RUN_DIR="${RUN_DIR:-$ROOT/experiments/runs/$PHASE/$DATASET/$ALGORITHM/$RUN_ID}"
export PHASE DATASET ALGORITHM RUN_ID MODEL ROUNDS LOCAL_EPOCHS LR NUM_CLIENTS JOIN_RATIO EVAL_GAP DEVICE_ID SEED BATCH_SIZE PYTHON_BIN GOAL RUN_DIR

# Resumability: a finished run (status.json == completed) is skipped so a phase
# can be re-launched after a crash; an incomplete/partial run dir is wiped and
# redone. FORCE=1 always wipes. SKIP_EXISTING=0 restores strict error behavior.
SKIP_EXISTING="${SKIP_EXISTING:-1}"
if [[ "${FORCE:-0}" == "1" ]]; then
  rm -rf "$RUN_DIR"
elif [[ -e "$RUN_DIR" ]]; then
  if [[ "$SKIP_EXISTING" == "1" ]] && grep -q '"status": "completed"' "$RUN_DIR/status.json" 2>/dev/null; then
    echo "[run_one] skip (already completed): $RUN_DIR"
    exit 0
  fi
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    echo "[run_one] re-running incomplete run: $RUN_DIR" >&2
    rm -rf "$RUN_DIR"
  else
    echo "Run directory already exists: $RUN_DIR" >&2
    echo "Set FORCE=1 to overwrite, or SKIP_EXISTING=1 to skip/redo." >&2
    exit 2
  fi
fi
mkdir -p "$RUN_DIR/results" "$RUN_DIR/models"

{
  echo "ROOT=$ROOT"
  echo "PHASE=$PHASE"
  echo "DATASET=$DATASET"
  echo "ALGORITHM=$ALGORITHM"
  echo "RUN_ID=$RUN_ID"
  echo "MODEL=$MODEL"
  echo "ROUNDS=$ROUNDS"
  echo "LOCAL_EPOCHS=$LOCAL_EPOCHS"
  echo "LR=$LR"
  echo "NUM_CLIENTS=$NUM_CLIENTS"
  echo "JOIN_RATIO=$JOIN_RATIO"
  echo "EVAL_GAP=$EVAL_GAP"
  echo "DEVICE_ID=$DEVICE_ID"
  echo "SEED=$SEED"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "GOAL=$GOAL"
  echo "PYTHON_BIN=$PYTHON_BIN"
  env | sort
} > "$RUN_DIR/env.txt"

git -C "$ROOT" status --short > "$RUN_DIR/git_status.txt" || true
if [[ -f "$ROOT/dataset/$DATASET/config.json" ]]; then
  cp "$ROOT/dataset/$DATASET/config.json" "$RUN_DIR/dataset_config.json"
fi

python3 - "$RUN_DIR/metadata.json" "$@" <<'PY'
import json, os, sys, time
path = sys.argv[1]
extra = sys.argv[2:]
keys = [
    "PHASE", "DATASET", "ALGORITHM", "RUN_ID", "MODEL", "ROUNDS", "LOCAL_EPOCHS",
    "LR", "NUM_CLIENTS", "JOIN_RATIO", "EVAL_GAP", "DEVICE_ID", "SEED", "BATCH_SIZE", "GOAL",
]
meta = {k.lower(): os.environ.get(k) for k in keys}
meta["created_at_unix"] = time.time()
meta["extra_args"] = extra
with open(path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, sort_keys=True)
PY

cmd=("$PYTHON_BIN" -u main.py
  -data "$DATASET"
  -m "$MODEL"
  -algo "$ALGORITHM"
  -gr "$ROUNDS"
  -ls "$LOCAL_EPOCHS"
  -lr "$LR"
  -nc "$NUM_CLIENTS"
  -jr "$JOIN_RATIO"
  -eg "$EVAL_GAP"
  -did "$DEVICE_ID"
  -lbs "$BATCH_SIZE"
  -go "$GOAL"
  --seed "$SEED"
  --results_save_path "$RUN_DIR/results"
  --model_save_path "$RUN_DIR/models"
  "$@"
)
printf '%q ' "${cmd[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"

cd "$ROOT/system"
set +e

# Stall watchdog: a wedged GPU (seen on this WSL2/Blackwell box) pins the device
# at 100% with no progress and never exits, so a plain wait() would hang the whole
# campaign. We heartbeat on stdout.log mtime; if it goes silent for STALL_TIMEOUT
# while the process is alive, we kill the run, retry inline up to MAX_STALL_RETRIES,
# and if it still stalls we leave the run incomplete (status != completed) and let
# the phase continue — a later relaunch redoes it via resumability. Genuine crashes
# (non-zero exit, NaN) are untouched and still surface.
WATCHDOG="${WATCHDOG:-1}"
STALL_TIMEOUT="${STALL_TIMEOUT:-600}"      # seconds of no stdout output => treat as hung
WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-30}"
MAX_STALL_RETRIES="${MAX_STALL_RETRIES:-2}"

run_attempt() {
  : > "$RUN_DIR/stdout.log"; : > "$RUN_DIR/stderr.log"
  "${cmd[@]}" > "$RUN_DIR/stdout.log" 2> "$RUN_DIR/stderr.log" &
  local pid=$! stalled=0 now mt age
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$WATCHDOG_INTERVAL"
    [[ "$WATCHDOG" == "1" ]] || continue
    now=$(date +%s)
    mt=$(stat -c %Y "$RUN_DIR/stdout.log" 2>/dev/null || echo "$now")
    age=$(( now - mt ))
    if (( age >= STALL_TIMEOUT )); then
      echo "[run_one] WATCHDOG: no output for ${age}s (>${STALL_TIMEOUT}s) — killing hung run $pid" >&2
      kill -9 "$pid" 2>/dev/null
      pkill -9 -P "$pid" 2>/dev/null
      stalled=1
      break
    fi
  done
  wait "$pid" 2>/dev/null
  local rc=$?
  (( stalled )) && return 124
  return "$rc"
}

status=0
attempt=0
while :; do
  run_attempt
  status=$?
  [[ "$status" -ne 124 ]] && break          # natural exit (ok or real error): done
  attempt=$(( attempt + 1 ))
  if (( attempt > MAX_STALL_RETRIES )); then
    echo "[run_one] WATCHDOG: gave up after ${attempt} stalls, leaving incomplete: $RUN_DIR" >&2
    break
  fi
  echo "[run_one] WATCHDOG: retry ${attempt}/${MAX_STALL_RETRIES} after stall" >&2
  sleep 5
done
set -e
cat "$RUN_DIR/stdout.log" "$RUN_DIR/stderr.log" > "$RUN_DIR/terminal.log" 2>/dev/null || true

# Derive the per-run artifact set. Runs unconditionally (even on failure) so
# status.json captures crashes. Uses PYTHON_BIN so torch/GPU manifest fields and
# matplotlib plots populate; never fails the run.
"$PYTHON_BIN" "$ROOT/experiments/finalize_run.py" "$RUN_DIR" "$ALGORITHM" "$status" || \
  echo "[run_one] finalize_run.py failed (non-fatal)" >&2

if [[ "$status" -eq 0 ]]; then
  python3 "$ROOT/experiments/validate_run.py" "$RUN_DIR" "$ALGORITHM"
  exit 0
fi
# Watchdog give-up (124): run left incomplete on purpose — don't abort the phase,
# a relaunch redoes it. Any other non-zero is a genuine failure and propagates.
if [[ "$status" -eq 124 ]]; then
  exit 0
fi
exit "$status"
