#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/content/whisper-pularr}"
RUNS_ROOT="${2:-/content/drive/MyDrive/omnilingual-pularr-runs}"
HF_HOME="${HF_HOME:-/content/hf-cache}"

export HF_HOME
export PYTHONUNBUFFERED=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -d "$ROOT" ]]; then
  echo "Repository root not found: $ROOT" >&2
  exit 1
fi

mkdir -p "$HF_HOME" "$RUNS_ROOT"
cd "$ROOT"

python "$ROOT/colab/run_omnilingual_t4_free.py" bootstrap \
  --runs-root "$RUNS_ROOT" \
  --hf-cache "$HF_HOME"
