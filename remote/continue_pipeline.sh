#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${1:-/root/whisper-pularr}"
DATASET_NAME="${2:-google/WaxalNLP}"
DATASET_CONFIG="${3:-ful_asr}"
MODEL_ID="${4:-openai/whisper-small}"
TEACHER_MODEL="${5:-openai/whisper-large-v3}"
WHISPER_LANGUAGE="${6:-}"
SKIP_ZERO_SHOT="${SKIP_ZERO_SHOT:-1}"

source "$REMOTE_ROOT/.venv/bin/activate"
cd "$REMOTE_ROOT"
export HF_HOME="$REMOTE_ROOT/hf-cache"
export TRANSFORMERS_CACHE="$REMOTE_ROOT/hf-cache/transformers"

mkdir -p "$REMOTE_ROOT/logs" "$REMOTE_ROOT/reports" "$REMOTE_ROOT/artifacts" "$REMOTE_ROOT/runs"
WHISPER_LANGUAGE_ARGS=()
if [[ -n "$WHISPER_LANGUAGE" ]]; then
  WHISPER_LANGUAGE_ARGS=(--whisper-language "$WHISPER_LANGUAGE")
fi

PSEUDO_REPORT_PATH="$REMOTE_ROOT/artifacts/pseudo_labels.report.json"
if [[ ! -f "$PSEUDO_REPORT_PATH" && -f "$REMOTE_ROOT/artifacts/pseudo_labels.jsonl.report.json" ]]; then
  PSEUDO_REPORT_PATH="$REMOTE_ROOT/artifacts/pseudo_labels.jsonl.report.json"
fi

LOCK_FILE="$REMOTE_ROOT/.continue_pipeline.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another continue_pipeline.sh instance is already running; exiting."
  exit 0
fi

wait_for_process() {
  local pattern="$1"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    sleep 60
  done
}

wait_for_gpu_idle() {
  while pgrep -f "python (train|evaluate|pseudo_label)\.py" >/dev/null 2>&1; do
    sleep 60
  done
}

run_trial() {
  local preset="$1"
  local run_dir="$REMOTE_ROOT/runs/$preset"
  local pattern="python train.py --stage supervised --preset $preset"
  if [[ -f "$run_dir/run_summary.json" ]]; then
    return 0
  fi
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    wait_for_process "$pattern"
  else
    python train.py \
      --stage supervised \
      --preset "$preset" \
      --dataset-name "$DATASET_NAME" \
      --dataset-config "$DATASET_CONFIG" \
      --model-id "$MODEL_ID" \
      "${WHISPER_LANGUAGE_ARGS[@]}" \
      --output-dir "$run_dir"
  fi
  [[ -f "$run_dir/run_summary.json" ]]
}

if [[ "$SKIP_ZERO_SHOT" != "1" && ! -f "$REMOTE_ROOT/reports/zero_shot_small_validation.summary.json" ]]; then
  wait_for_gpu_idle
  python evaluate.py \
    --checkpoint "$MODEL_ID" \
    --dataset-name "$DATASET_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    "${WHISPER_LANGUAGE_ARGS[@]}" \
    --split validation \
    --output-path "$REMOTE_ROOT/reports/zero_shot_small_validation.json"
fi

if [[ "$SKIP_ZERO_SHOT" != "1" && ! -f "$REMOTE_ROOT/reports/zero_shot_teacher_validation.summary.json" ]]; then
  wait_for_gpu_idle
  python evaluate.py \
    --checkpoint "$TEACHER_MODEL" \
    --dataset-name "$DATASET_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    "${WHISPER_LANGUAGE_ARGS[@]}" \
    --split validation \
    --output-path "$REMOTE_ROOT/reports/zero_shot_teacher_validation.json"
fi

run_trial trial_a
run_trial trial_b
run_trial trial_c

SELF_TRAIN_ROOT="$REMOTE_ROOT/runs/self_train_snapshots"
SELF_TRAIN_SEQUENCE_SUMMARY="$SELF_TRAIN_ROOT/sequence_summary.json"
SELF_TRAIN_BASE_CHECKPOINT="$(python - <<PY
from pathlib import Path

from whisper_pularr.pipeline_status import select_best_supervised_checkpoint

runs_root = Path("$REMOTE_ROOT") / "runs"
checkpoint = select_best_supervised_checkpoint(runs_root)
if checkpoint:
    print(checkpoint)
else:
    fallback = Path("$REMOTE_ROOT") / "downloads" / "trial_a_best_full_eval" / "best_full_eval"
    print(str(fallback))
PY
)"
SELF_TRAIN_SEQUENCE_COMPLETE="$(python - <<PY
from pathlib import Path

from whisper_pularr.pipeline_status import sequence_summary_complete

summary_path = Path("$SELF_TRAIN_SEQUENCE_SUMMARY")
manifests_dir = Path("$REMOTE_ROOT") / "artifacts" / "pseudo_labels_manifests"
print("1" if sequence_summary_complete(summary_path, manifests_dir) else "0")
PY
)"
SELF_TRAIN_BASE_ARGS=()
if [[ -d "$SELF_TRAIN_BASE_CHECKPOINT" ]]; then
  SELF_TRAIN_BASE_ARGS=(--base-checkpoint "$SELF_TRAIN_BASE_CHECKPOINT")
fi
if [[ "$SELF_TRAIN_SEQUENCE_COMPLETE" != "1" ]]; then
  if [[ ! -f "$PSEUDO_REPORT_PATH" ]]; then
    if ! pgrep -f "python run_self_train_sequence.py" >/dev/null 2>&1; then
      python run_self_train_sequence.py \
        --dataset-name "$DATASET_NAME" \
        --dataset-config "$DATASET_CONFIG" \
        --model-id "$MODEL_ID" \
        "${WHISPER_LANGUAGE_ARGS[@]}" \
        --manifests-dir "$REMOTE_ROOT/artifacts/pseudo_labels_manifests" \
        --output-root "$SELF_TRAIN_ROOT" \
        --watch \
        --poll-seconds 10 \
        "${SELF_TRAIN_BASE_ARGS[@]}" &
    fi
    if ! pgrep -f "python pseudo_label.py" >/dev/null 2>&1; then
      wait_for_gpu_idle
      python pseudo_label.py \
        --dataset-name "$DATASET_NAME" \
        --dataset-config "$DATASET_CONFIG" \
        --teacher-model "$TEACHER_MODEL" \
        "${WHISPER_LANGUAGE_ARGS[@]}" \
        --output-path "$REMOTE_ROOT/artifacts/pseudo_labels.jsonl"
    fi
    wait_for_process "python run_self_train_sequence.py"
  else
    if pgrep -f "python run_self_train_sequence.py" >/dev/null 2>&1; then
      wait_for_process "python run_self_train_sequence.py"
    else
      python run_self_train_sequence.py \
        --dataset-name "$DATASET_NAME" \
        --dataset-config "$DATASET_CONFIG" \
        --model-id "$MODEL_ID" \
        "${WHISPER_LANGUAGE_ARGS[@]}" \
        --manifests-dir "$REMOTE_ROOT/artifacts/pseudo_labels_manifests" \
        --output-root "$SELF_TRAIN_ROOT" \
        "${SELF_TRAIN_BASE_ARGS[@]}"
    fi
  fi
fi

LAST_BEST_FULL_EVAL="$(python - <<PY
import json
from pathlib import Path

summary_path = Path("$SELF_TRAIN_SEQUENCE_SUMMARY")
with summary_path.open("r", encoding="utf-8") as handle:
    summary = json.load(handle)
print(summary["last_best_full_eval_dir"])
PY
)"

if [[ ! -f "$REMOTE_ROOT/reports/final_test_eval.summary.json" ]]; then
  wait_for_gpu_idle
  python evaluate.py \
    --checkpoint "$LAST_BEST_FULL_EVAL" \
    --dataset-name "$DATASET_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    "${WHISPER_LANGUAGE_ARGS[@]}" \
    --split test \
    --output-path "$REMOTE_ROOT/reports/final_test_eval.json"
fi
