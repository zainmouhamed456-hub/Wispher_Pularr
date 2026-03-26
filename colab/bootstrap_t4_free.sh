#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
