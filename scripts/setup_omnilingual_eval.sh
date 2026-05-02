#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${OMNI_VENV_DIR:-"$ROOT_DIR/.venv-omni"}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "$(uname -s)" in
  Linux*) ;;
  *)
    cat >&2 <<'MSG'
Omnilingual ASR cannot be installed in this native Windows shell.

Its fairseq2 dependency needs fairseq2n native wheels. Meta publishes the
compatible fairseq2n wheels for Linux, but not for win_amd64.

Use this script from WSL/Linux instead:
  bash scripts/setup_omnilingual_eval.sh
MSG
    exit 1
    ;;
esac

"$PYTHON_BIN" - <<'PY'
import sys

version = sys.version_info
if version < (3, 10) or version >= (3, 13):
    raise SystemExit(
        "Omnilingual ASR needs Python >=3.10,<3.13 for the fairseq2 wheel stack; "
        f"found {sys.version.split()[0]}"
    )
PY

if command -v apt-get >/dev/null 2>&1; then
  missing_packages=()
  if ! ldconfig -p 2>/dev/null | grep -q 'libsndfile\.so'; then
    missing_packages+=(libsndfile1)
  fi
  if ! command -v ffmpeg >/dev/null 2>&1; then
    missing_packages+=(ffmpeg)
  fi
  if ((${#missing_packages[@]})); then
    sudo apt-get update
    sudo apt-get install -y "${missing_packages[@]}"
  fi
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements-omnilingual-linux.txt"

python "$ROOT_DIR/scripts/verify_omnilingual_import.py"

cat <<MSG

Ready. Activate with:
  source "$VENV_DIR/bin/activate"
MSG
