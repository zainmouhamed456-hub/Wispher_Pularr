#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${1:-$HOME/whisper-pularr}"
VENV_PATH="${2:-$REMOTE_ROOT/.venv}"
TORCH_INDEX_URL="${3:-https://download.pytorch.org/whl/cu124}"

mkdir -p "$REMOTE_ROOT"
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "$TORCH_INDEX_URL" torch torchaudio
python -m pip install -r "$REMOTE_ROOT/requirements.txt"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "warning: ffmpeg is not installed on the remote host; openai-whisper may fail until ffmpeg is available" >&2
fi
