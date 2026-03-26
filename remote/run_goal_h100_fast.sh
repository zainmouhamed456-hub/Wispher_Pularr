#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${1:-/root/whisper-pularr}"
DATASET_NAME="${2:-google/WaxalNLP}"
DATASET_CONFIG="${3:-ful_asr}"
MODEL_ID="${4:-openai/whisper-small}"
TEACHER_MODEL="${5:-openai/whisper-large-v3}"
WHISPER_LANGUAGE="${6:-}"
AUX_DATASET_NAME="${7:-}"
AUX_DATASET_CONFIG="${8:-}"

source "$REMOTE_ROOT/.venv/bin/activate"
cd "$REMOTE_ROOT"
export HF_HOME="$REMOTE_ROOT/hf-cache"
export TRANSFORMERS_CACHE="$REMOTE_ROOT/hf-cache/transformers"

mkdir -p "$REMOTE_ROOT/logs" "$REMOTE_ROOT/artifacts" "$REMOTE_ROOT/reports" "$REMOTE_ROOT/runs"

WHISPER_LANGUAGE_ARGS=()
if [[ -n "$WHISPER_LANGUAGE" ]]; then
  WHISPER_LANGUAGE_ARGS=(--whisper-language "$WHISPER_LANGUAGE")
fi

AUX_ARGS=()
if [[ -n "$AUX_DATASET_NAME" && -n "$AUX_DATASET_CONFIG" ]]; then
  AUX_ARGS=(--aux-dataset-name "$AUX_DATASET_NAME" --aux-dataset-config "$AUX_DATASET_CONFIG" --aux-labeled-repeat-count 1)
fi

BEST_BASE_CHECKPOINT="$(python - <<PY
from pathlib import Path

from whisper_pularr.pipeline_status import select_best_supervised_checkpoint

runs_root = Path("$REMOTE_ROOT") / "runs"
checkpoint = select_best_supervised_checkpoint(runs_root, trial_names=("goal_trial_a", "trial_a", "trial_b", "trial_c"))
if checkpoint:
    print(checkpoint)
else:
    fallback = Path("$REMOTE_ROOT") / "downloads" / "trial_a_best_full_eval" / "best_full_eval"
    if fallback.is_dir():
        print(str(fallback))
PY
)"

SUPERVISED_RUN_DIR="$REMOTE_ROOT/runs/goal_trial_a"
if [[ ! -f "$SUPERVISED_RUN_DIR/run_summary.json" ]]; then
  python train.py \
    --stage supervised \
    --preset trial_a \
    --dataset-name "$DATASET_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    --model-id "$MODEL_ID" \
    "${WHISPER_LANGUAGE_ARGS[@]}" \
    "${AUX_ARGS[@]}" \
    --num-train-epochs 8 \
    --early-stop-patience-epochs 2 \
    --save-total-limit 1 \
    --output-dir "$SUPERVISED_RUN_DIR"
fi

SUPERVISED_CHECKPOINT="$SUPERVISED_RUN_DIR/best_full_eval"
if [[ ! -d "$SUPERVISED_CHECKPOINT" && -n "$BEST_BASE_CHECKPOINT" && -d "$BEST_BASE_CHECKPOINT" ]]; then
  SUPERVISED_CHECKPOINT="$BEST_BASE_CHECKPOINT"
fi

python evaluate.py \
  --checkpoint "$SUPERVISED_CHECKPOINT" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  --split validation \
  --output-path "$REMOTE_ROOT/reports/goal_supervised_validation.json"

python pseudo_label.py \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --teacher-model "$TEACHER_MODEL" \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  --beam-size 5 \
  --max-keep-multiple 1.0 \
  --candidate-pool-multiple 1.0 \
  --manifest-every 5000 \
  --output-path "$REMOTE_ROOT/artifacts/pseudo_labels.jsonl"

python run_self_train_sequence.py \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --model-id "$MODEL_ID" \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  --base-checkpoint "$SUPERVISED_CHECKPOINT" \
  --manifests-dir "$REMOTE_ROOT/artifacts/pseudo_labels_manifests" \
  --output-root "$REMOTE_ROOT/runs/self_train_fast" \
  --final-only

LAST_BEST_FULL_EVAL="$(python - <<PY
import json
from pathlib import Path

summary_path = Path("$REMOTE_ROOT/runs/self_train_fast/sequence_summary.json")
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
  --split validation \
  --output-path "$REMOTE_ROOT/reports/goal_final_validation.json"

python compare_checkpoints.py \
  --checkpoint "$MODEL_ID" \
  --checkpoint "$SUPERVISED_CHECKPOINT" \
  --checkpoint "$LAST_BEST_FULL_EVAL" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  --skip-full \
  --output-dir "$REMOTE_ROOT/reports/goal_comparison"
