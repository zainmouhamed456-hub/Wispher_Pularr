# Colab T4 Free Runbook

This runbook matches the promotion-aware Colab flow in `colab/run_t4_free.py`.

Local status before Colab:
- The bundled seed checkpoint is `downloads/trial_a_best_full_eval/best_full_eval`.
- Local direct-parquet spot checks show the finetuned checkpoint is much better than zero-shot `openai/whisper-small`.
- On the local 10-sample slice, beam `3` changed one prediction, kept WER flat, and slightly worsened CER.
- On the local 32-sample CPU slice, beam `1` reached normalized WER `43.04%` and normalized CER `11.41%`.
- Even with that signal, Session 1 in Colab should still rerun the fixed 64-sample beam sweep and then do full validation with the selected beam.

## Session 1: Decode Selection And Baseline Eval

Use these settings in the notebook config cell:

```bash
COLAB_EVAL_ONLY="1"
COLAB_BASE_MODEL=""
COLAB_RESUME_FROM=""
COLAB_COMPARE_BEAMS="1,3"
COLAB_FIXED_SLICE_SIZE="64"
COLAB_RUN_SELF_TRAIN="0"
COLAB_MAX_TRAIN_SAMPLES="1024"
COLAB_MAX_EVAL_SAMPLES="64"
```

Expected behavior:
- Compares `openai/whisper-small` and the bundled best checkpoint on the same 64-sample validation slice.
- Tries beams `1` and `3`.
- Writes decode selection to `RUNS_ROOT/reports/colab_decode_selection.json`.
- Writes the promoted baseline to `RUNS_ROOT/reports/colab_promotion_summary.json`.
- Runs `analyze_eval.py` on the promoted full validation output.

Inspect after Session 1:
- `RUNS_ROOT/reports/colab_decode_selection.json`
- `RUNS_ROOT/reports/colab_promotion_summary.json`
- `RUNS_ROOT/reports/colab_base_eval_analysis.json`

## Session 2: First Supervised Continuation

Use these settings:

```bash
COLAB_EVAL_ONLY="0"
COLAB_BASE_MODEL=""
COLAB_RESUME_FROM=""
COLAB_COMPARE_BEAMS="1,3"
COLAB_FIXED_SLICE_SIZE="64"
COLAB_SUPERVISED_EPOCHS="1"
COLAB_EARLY_STOP_PATIENCE="1"
COLAB_MAX_TRAIN_SAMPLES="1024"
COLAB_MAX_EVAL_SAMPLES="64"
COLAB_RUN_SELF_TRAIN="0"
```

Expected behavior:
- Starts from the promoted checkpoint if one exists.
- Runs supervised `trial_a` continuation with streaming.
- Evaluates the resulting `best_full_eval` using the selected decode beam from Session 1.
- Promotes the new checkpoint only if normalized WER improves by at least `1.0` absolute point, or if WER ties and CER improves.

Inspect after Session 2:
- `RUNS_ROOT/runs/colab_supervised/session_*/promotion_decision.json`
- `RUNS_ROOT/reports/colab_promotion_summary.json`

## Session 3: Larger Supervised Continuation

Repeat Session 2, but raise the training cap:

```bash
COLAB_EVAL_ONLY="0"
COLAB_BASE_MODEL=""
COLAB_RESUME_FROM=""
COLAB_COMPARE_BEAMS="1,3"
COLAB_FIXED_SLICE_SIZE="64"
COLAB_SUPERVISED_EPOCHS="1"
COLAB_EARLY_STOP_PATIENCE="1"
COLAB_MAX_TRAIN_SAMPLES="4096"
COLAB_MAX_EVAL_SAMPLES="64"
COLAB_RUN_SELF_TRAIN="0"
```

If the runtime disconnects but you have a saved trainer checkpoint, set:

```bash
COLAB_RESUME_FROM="/content/drive/MyDrive/whisper-pularr-runs/runs/colab_supervised/session_XXX/checkpoint-YYY"
```

Otherwise leave `COLAB_RESUME_FROM` empty and let the launcher continue from the last promoted `best_full_eval`.

## Optional Session 4: Auxiliary Supervised Continuation

Only do this if you have a valid auxiliary dataset name and config.

Use the same settings as Session 3, plus:

```bash
AUX_DATASET_NAME="your/aux-dataset"
AUX_DATASET_CONFIG="your_config"
```

The launcher already applies `--aux-labeled-repeat-count 1`.

## Stage 2: Pseudo-Labeling And Self-Training

Only start this after the best supervised checkpoint is stable.

```bash
COLAB_EVAL_ONLY="0"
COLAB_BASE_MODEL=""
COLAB_RESUME_FROM=""
COLAB_COMPARE_BEAMS="1,3"
COLAB_FIXED_SLICE_SIZE="64"
COLAB_RUN_SELF_TRAIN="1"
COLAB_PSEUDO_BEAM_SIZE="3"
COLAB_PSEUDO_BATCH_SIZE="2"
COLAB_MAX_KEEP_MULTIPLE="0.75"
COLAB_MANIFEST_EVERY="4000"
```

Expected behavior:
- Generates pseudo-labels with the current quality thresholds.
- Runs `run_self_train_sequence.py --final-only` from the promoted supervised checkpoint.
- Evaluates the final checkpoint with the selected decode beam.
- Promotes only if it beats the current best under the same promotion rule.

Inspect after Stage 2:
- `RUNS_ROOT/artifacts/colab_pseudo_labels.jsonl`
- `RUNS_ROOT/runs/colab_self_train/sequence_summary.json`
- `RUNS_ROOT/reports/colab_self_train_validation.analysis.json`
- `RUNS_ROOT/reports/colab_promotion_summary.json`

## Error Analysis Guidance

After each promoted full evaluation:
- Review the worst `50` to `100` samples in the matching `*.analysis.json` file.
- If repeated spacing, apostrophe, or hyphen mismatches show up, update `whisper_pularr/text.py` and rerun the same comparison.
- If pseudo-label reports show repeated `language_mismatch`, tighten `min_labeled_token_ratio` before generating another manifest.
