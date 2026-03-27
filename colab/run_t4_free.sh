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
export RUNS_ROOT

python "$ROOT/colab/run_t4_free.py" \
  "$ROOT" \
  "$DATASET_NAME" \
  "$DATASET_CONFIG" \
  "$MODEL_ID" \
  "$TEACHER_MODEL" \
  "$WHISPER_LANGUAGE" \
  "$AUX_DATASET_NAME" \
  "$AUX_DATASET_CONFIG"
