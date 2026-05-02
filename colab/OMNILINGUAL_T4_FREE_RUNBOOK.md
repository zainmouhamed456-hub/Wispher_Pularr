# Omnilingual Pularr Colab T4 Runbook

This runbook fine-tunes Meta Omnilingual ASR CTC 300M on `google/WaxalNLP/ful_asr`
from a Google Colab Free T4 runtime. It uses Drive for checkpoints and promotes a
trained checkpoint only when normalized validation WER improves, with normalized
CER as the tie-breaker.

## Runtime Setup

Use a Colab GPU runtime:

```bash
Runtime > Change runtime type > T4 GPU
```

Mount Drive and clone the repo:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
test -d /content/whisper-pularr || git clone https://github.com/zainmouhamed456-hub/Wispher_Pularr.git /content/whisper-pularr
cd /content/whisper-pularr
```

Install Omnilingual and CUDA dependencies:

```bash
bash colab/bootstrap_omnilingual_t4_free.sh \
  /content/whisper-pularr \
  /content/drive/MyDrive/omnilingual-pularr-runs
```

Or directly:

```bash
python colab/run_omnilingual_t4_free.py bootstrap \
  --runs-root /content/drive/MyDrive/omnilingual-pularr-runs
```

If a notebook cell raises `ModuleNotFoundError: No module named
'omnilingual_asr'`, run the install/verify cell in
`colab/Omnilingual_Pularr_T4_Free.ipynb`. It is standalone: it bootstraps the
dependency if needed and adds Meta's source directory to the current notebook
kernel.

```python
%cd /content/whisper-pularr
!python colab/run_omnilingual_t4_free.py bootstrap --runs-root /content/drive/MyDrive/omnilingual-pularr-runs

import importlib, sys
from pathlib import Path
external_root = Path("/content/drive/MyDrive/omnilingual-pularr-runs/external/omnilingual-asr")
for path in (external_root / "src", external_root):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
importlib.invalidate_caches()

import torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
print("cuda_available=", torch.cuda.is_available())
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("pipeline=", ASRInferencePipeline.__name__)
```

## Session 0: Baseline Selection

```bash
python colab/run_omnilingual_t4_free.py baseline \
  --dataset-name google/WaxalNLP \
  --dataset-config ful_asr \
  --lang ful_Latn \
  --runs-root /content/drive/MyDrive/omnilingual-pularr-runs
```

Expected output:

- `reports/omnilingual_baseline_summary.json`
- `reports/omnilingual_promotion_summary.json`

The script compares `omniASR_CTC_300M` and `omniASR_CTC_300M_v2` on the same
64 validation examples. If one card is unavailable in the installed
Omnilingual package, the failure is recorded and the available candidate can
still win.

## Session 1: Dataset Prep

```bash
python colab/run_omnilingual_t4_free.py prepare \
  --dataset-name google/WaxalNLP \
  --dataset-config ful_asr \
  --lang ful_Latn \
  --runs-root /content/drive/MyDrive/omnilingual-pularr-runs
```

Expected output:

- `artifacts/waxal_ful_pularr/version=0/corpus=waxal/split=train/language=ful_Latn/*.parquet`
- `artifacts/waxal_ful_pularr/version=0/corpus=waxal/split=dev/language=ful_Latn/*.parquet`
- `artifacts/waxal_ful_pularr/version=0/corpus=waxal/split=test/language=ful_Latn/*.parquet`
- `artifacts/waxal_ful_pularr/language_distribution_0.tsv`
- `artifacts/omnilingual_generated/cards/datasets/waxal_ful_pularr.yaml`
- `artifacts/omnilingual_generated/configs/ctc_finetune_pularr.yaml`
- `artifacts/omnilingual_generated/configs/ctc_eval_pularr.yaml`

Waxal `validation` is mapped to Omnilingual `dev`.

## Session 2: Smoke Fine-Tune

```bash
python colab/run_omnilingual_t4_free.py train \
  --runs-root /content/drive/MyDrive/omnilingual-pularr-runs \
  --smoke-steps 100
```

This proves that the official Meta recipe, dataset card, parquet dataset, and
checkpoint output all work before spending a long Colab session.

Use `--skip-train` to only generate configs.

## Session 3: Main Fine-Tune

```bash
python colab/run_omnilingual_t4_free.py train \
  --runs-root /content/drive/MyDrive/omnilingual-pularr-runs \
  --main-steps 2000 \
  --learning-rate 1e-5 \
  --grad-accumulation 16 \
  --validate-every 250 \
  --checkpoint-every 500
```

Expected output:

- `runs/omnilingual_ctc_session_*/`
- `runs/omnilingual_ctc_session_*/checkpoints/`
- validation transcriptions under the recipe output when validation runs

If Colab disconnects, rerun the same command. The official fairseq2 recipe
manages its own checkpoint files under the run directory.

## Eval And Promotion

After a train run produces `transcriptions/*.ref.txt` and `transcriptions/*.hyp.txt`:

```bash
python colab/run_omnilingual_t4_free.py eval \
  --runs-root /content/drive/MyDrive/omnilingual-pularr-runs \
  --eval-dir /content/drive/MyDrive/omnilingual-pularr-runs/runs/omnilingual_ctc_session_001
```

Then promote only if validation WER improves:

```bash
python colab/run_omnilingual_t4_free.py promote \
  --runs-root /content/drive/MyDrive/omnilingual-pularr-runs \
  --candidate-checkpoint /content/drive/MyDrive/omnilingual-pularr-runs/runs/omnilingual_ctc_session_001
```

Expected output:

- `reports/omnilingual_validation_eval.json`
- `reports/omnilingual_promotion_decision.json`
- `reports/omnilingual_promotion_summary.json`

## One-Command Flow

After dependencies are installed, this runs baseline, prep, smoke training, and
main training:

```bash
python colab/run_omnilingual_t4_free.py all \
  --dataset-name google/WaxalNLP \
  --dataset-config ful_asr \
  --lang ful_Latn \
  --runs-root /content/drive/MyDrive/omnilingual-pularr-runs
```

If the final recipe output includes validation transcriptions, `all` also writes
`omnilingual_validation_eval.json` and applies the promotion rule.

## Useful Overrides

```bash
export HF_HOME=/content/hf-cache
export OMNI_LANG=ful_Latn
export OMNI_MAX_DURATION_SECONDS=40
export OMNI_BASELINE_SAMPLES=64
export OMNI_MAIN_STEPS=2000
export OMNI_SMOKE_STEPS=100
```

For quick debugging:

```bash
python colab/run_omnilingual_t4_free.py prepare \
  --max-samples-per-split 8 \
  --runs-root /content/drive/MyDrive/omnilingual-pularr-runs
```
