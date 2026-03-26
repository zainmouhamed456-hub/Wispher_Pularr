#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${1:-/root/whisper-pularr}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8787}"
REFRESH_SECONDS="${4:-20}"

source "$REMOTE_ROOT/.venv/bin/activate"
cd "$REMOTE_ROOT"

python dashboard.py \
  --root "$REMOTE_ROOT" \
  --host "$HOST" \
  --port "$PORT" \
  --refresh-seconds "$REFRESH_SECONDS"
