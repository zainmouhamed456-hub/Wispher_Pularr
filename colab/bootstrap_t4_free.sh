#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  accelerate==1.2.1 \
  "datasets[audio]==3.2.0" \
  evaluate==0.4.3 \
  jiwer==3.0.5 \
  openai-whisper==20250625 \
  psutil==6.1.1 \
  sentencepiece==0.2.0 \
  transformers==4.47.1
