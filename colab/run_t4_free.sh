#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/content/whisper-pularr}"
DATASET_NAME="${2:-google/WaxalNLP}"
DATASET_CONFIG="${3:-ful_asr}"
MODEL_ID="${4:-openai/whisper-small}"
TEACHER_MODEL="${5:-openai/whisper-large-v3}"
WHISPER_LANGUAGE="${6:-}"
AUX_DATASET_NAME="${7:-}"
AUX_DATASET_CONFIG="${8:-}"

export HF_HOME="${HF_HOME:-/content/hf-cache}"
export PYTHONUNBUFFERED=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -d "$ROOT" ]]; then
  if [[ -d "/content/whisper-pularr" ]]; then
    ROOT="/content/whisper-pularr"
  elif [[ -f "./train.py" && -d "./colab" ]]; then
    ROOT="$(pwd)"
  else
    echo "Repository root not found: $ROOT" >&2
    echo "Run the clone cell first, or clone the repo to /content/whisper-pularr." >&2
    exit 1
  fi
fi

cd "$ROOT"
mkdir -p "$HF_HOME"

if [[ -d "/content/drive/MyDrive" ]]; then
  RUNS_ROOT="${RUNS_ROOT:-/content/drive/MyDrive/whisper-pularr-runs}"
else
  RUNS_ROOT="${RUNS_ROOT:-/content/whisper-pularr-runs}"
fi

mkdir -p "$RUNS_ROOT"/artifacts "$RUNS_ROOT"/reports "$RUNS_ROOT"/runs

WHISPER_LANGUAGE_ARGS=()
if [[ -n "$WHISPER_LANGUAGE" ]]; then
  WHISPER_LANGUAGE_ARGS=(--whisper-language "$WHISPER_LANGUAGE")
fi

AUX_ARGS=()
if [[ -n "$AUX_DATASET_NAME" && -n "$AUX_DATASET_CONFIG" ]]; then
  AUX_ARGS=(--aux-dataset-name "$AUX_DATASET_NAME" --aux-dataset-config "$AUX_DATASET_CONFIG" --aux-labeled-repeat-count 1)
fi

python train.py \
  --stage supervised \
  --preset trial_a \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --model-id "$MODEL_ID" \
  --streaming \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  "${AUX_ARGS[@]}" \
  --num-train-epochs "${COLAB_SUPERVISED_EPOCHS:-3}" \
  --early-stop-patience-epochs "${COLAB_EARLY_STOP_PATIENCE:-1}" \
  --max-train-samples "${COLAB_MAX_TRAIN_SAMPLES:-3000}" \
  --max-eval-samples "${COLAB_MAX_EVAL_SAMPLES:-300}" \
  --save-total-limit 1 \
  --output-dir "$RUNS_ROOT/runs/colab_trial_a"

python evaluate.py \
  --checkpoint "$RUNS_ROOT/runs/colab_trial_a/best_full_eval" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  --split validation \
  --output-path "$RUNS_ROOT/reports/colab_supervised_validation.json"

if [[ "${COLAB_RUN_SELF_TRAIN:-0}" != "1" ]]; then
  echo "Supervised stage complete. Set COLAB_RUN_SELF_TRAIN=1 to continue with pseudo-labeling and self-training."
  exit 0
fi

python pseudo_label.py \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --teacher-model "$TEACHER_MODEL" \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  --beam-size "${COLAB_PSEUDO_BEAM_SIZE:-3}" \
  --batch-size "${COLAB_PSEUDO_BATCH_SIZE:-2}" \
  --max-keep-multiple "${COLAB_MAX_KEEP_MULTIPLE:-0.75}" \
  --candidate-pool-multiple 1.0 \
  --manifest-every "${COLAB_MANIFEST_EVERY:-4000}" \
  --output-path "$RUNS_ROOT/artifacts/colab_pseudo_labels.jsonl"

python run_self_train_sequence.py \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --model-id "$MODEL_ID" \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  --base-checkpoint "$RUNS_ROOT/runs/colab_trial_a/best_full_eval" \
  --manifests-dir "$RUNS_ROOT/artifacts/colab_pseudo_labels_manifests" \
  --output-root "$RUNS_ROOT/runs/colab_self_train" \
  --final-only

python compare_checkpoints.py \
  --checkpoint "$MODEL_ID" \
  --checkpoint "$RUNS_ROOT/runs/colab_trial_a/best_full_eval" \
  --checkpoint "$RUNS_ROOT/runs/colab_self_train/pseudo_labels_final/best_full_eval" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  "${WHISPER_LANGUAGE_ARGS[@]}" \
  --skip-full \
  --output-dir "$RUNS_ROOT/reports/colab_comparison"
