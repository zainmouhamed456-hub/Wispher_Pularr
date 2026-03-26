#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${1:-/root/whisper-pularr}"
DATASET_NAME="${2:-google/WaxalNLP}"
DATASET_CONFIG="${3:-ful_asr}"
MODEL_ID="${4:-openai/whisper-small}"
TEACHER_MODEL="${5:-openai/whisper-large-v3}"
WHISPER_LANGUAGE="${6:-}"
BEST_CHECKPOINT="${7:-}"
PSEUDO_MAX_KEEP_MULTIPLE="${8:-1.0}"
PSEUDO_CANDIDATE_POOL_MULTIPLE="${9:-1.1}"

source "$REMOTE_ROOT/.venv/bin/activate"
cd "$REMOTE_ROOT"
export HF_HOME="$REMOTE_ROOT/hf-cache"
export TRANSFORMERS_CACHE="$REMOTE_ROOT/hf-cache/transformers"

mkdir -p "$REMOTE_ROOT/logs" "$REMOTE_ROOT/artifacts" "$REMOTE_ROOT/reports" "$REMOTE_ROOT/runs"

BASE_CHECKPOINT_ARGS=()
if [[ -n "$BEST_CHECKPOINT" ]]; then
  BASE_CHECKPOINT_ARGS=(--base-checkpoint "$BEST_CHECKPOINT")
else
  BEST_CHECKPOINT="$(python - <<PY
from pathlib import Path

from whisper_pularr.pipeline_status import select_best_supervised_checkpoint

runs_root = Path("$REMOTE_ROOT") / "runs"
checkpoint = select_best_supervised_checkpoint(runs_root)
if checkpoint:
    print(checkpoint)
PY
)"
  if [[ -n "$BEST_CHECKPOINT" && -d "$BEST_CHECKPOINT" ]]; then
    BASE_CHECKPOINT_ARGS=(--base-checkpoint "$BEST_CHECKPOINT")
  elif [[ -d "$REMOTE_ROOT/downloads/trial_a_best_full_eval/best_full_eval" ]]; then
    BASE_CHECKPOINT_ARGS=(--base-checkpoint "$REMOTE_ROOT/downloads/trial_a_best_full_eval/best_full_eval")
  fi
fi
WHISPER_LANGUAGE_ARGS=()
if [[ -n "$WHISPER_LANGUAGE" ]]; then
  WHISPER_LANGUAGE_ARGS=(--whisper-language "$WHISPER_LANGUAGE")
fi

MANIFESTS_DIR="$REMOTE_ROOT/artifacts/pseudo_labels_manifests"
SELF_TRAIN_ROOT="$REMOTE_ROOT/runs/self_train_snapshots"

python run_self_train_sequence.py \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --model-id "$MODEL_ID" \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  --manifests-dir "$MANIFESTS_DIR" \
  --output-root "$SELF_TRAIN_ROOT" \
  --watch \
  --poll-seconds 10 \
  "${BASE_CHECKPOINT_ARGS[@]}" &
SEQUENCE_PID=$!

cleanup_sequence() {
  kill "$SEQUENCE_PID" >/dev/null 2>&1 || true
}
trap cleanup_sequence EXIT

python pseudo_label.py \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --teacher-model "$TEACHER_MODEL" \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  --max-keep-multiple "$PSEUDO_MAX_KEEP_MULTIPLE" \
  --candidate-pool-multiple "$PSEUDO_CANDIDATE_POOL_MULTIPLE" \
  --output-path "$REMOTE_ROOT/artifacts/pseudo_labels.jsonl"

wait "$SEQUENCE_PID"
trap - EXIT

LAST_BEST_FULL_EVAL="$(python - <<PY
import json
from pathlib import Path

summary_path = Path("$SELF_TRAIN_ROOT") / "sequence_summary.json"
with summary_path.open("r", encoding="utf-8") as handle:
    summary = json.load(handle)
print(summary["last_best_full_eval_dir"])
PY
)"

python evaluate.py \
  --checkpoint "$LAST_BEST_FULL_EVAL" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  --split test \
  --output-path "$REMOTE_ROOT/reports/final_test_eval.json"
