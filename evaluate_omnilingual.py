from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from whisper_pularr.data import load_asr_split, save_json
from whisper_pularr.eval_utils import compute_error_metrics
from whisper_pularr.settings import DEFAULT_DATASET_CONFIG, DEFAULT_DATASET_NAME
from whisper_pularr.text import normalize_transcript

DEFAULT_OMNI_MODEL_CARD = "omniASR_CTC_300M"
DEFAULT_OMNI_LANGUAGE = "ful_Latn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Meta Omnilingual ASR on WaxalNLP/Pularr.")
    parser.add_argument("--model-card", default=DEFAULT_OMNI_MODEL_CARD)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", choices=["validation", "test", "train"], default="validation")
    parser.add_argument("--lang", default=DEFAULT_OMNI_LANGUAGE)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def _require_omnilingual_pipeline():
    try:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing omnilingual-asr. On Linux/WSL run: bash scripts/setup_omnilingual_eval.sh"
        ) from exc
    return ASRInferencePipeline


def _audio_input(row: dict[str, Any]) -> dict[str, Any]:
    audio = row["audio"]
    return {
        "waveform": audio["array"],
        "sample_rate": int(audio.get("sampling_rate") or audio.get("sample_rate") or 16000),
    }


def evaluate_omnilingual_dataset(
    *,
    asr_pipeline: Any,
    dataset: Any,
    lang: str | None,
    batch_size: int,
    output_path: str | None = None,
) -> dict[str, Any]:
    rows = list(dataset)
    references: list[str] = []
    predictions: list[str] = []
    per_sample: list[dict[str, Any]] = []
    effective_batch_size = max(int(batch_size), 1)
    print(f"Evaluating {len(rows)} sample(s) with Omnilingual batch_size={effective_batch_size}...", flush=True)

    for start in range(0, len(rows), effective_batch_size):
        batch_rows = rows[start : start + effective_batch_size]
        batch_inputs = [_audio_input(row) for row in batch_rows]
        batch_lang = [lang] * len(batch_inputs) if lang else None
        batch_predictions = asr_pipeline.transcribe(batch_inputs, lang=batch_lang, batch_size=effective_batch_size)
        for row, prediction_value in zip(batch_rows, batch_predictions):
            prediction = str(prediction_value or "").strip()
            reference = str(row.get("transcription") or "").strip()
            references.append(reference)
            predictions.append(prediction)
            per_sample.append(
                {
                    "id": row.get("id"),
                    "reference": reference,
                    "prediction": prediction,
                    "normalized_reference": normalize_transcript(reference),
                    "normalized_prediction": normalize_transcript(prediction),
                }
            )
        print(f"Omnilingual eval progress: {len(per_sample)}/{len(rows)} sample(s)", flush=True)

    payload = {
        "sample_count": len(per_sample),
        "metrics": compute_error_metrics(references, predictions),
        "samples": per_sample,
    }
    if output_path:
        save_json(output_path, payload)
    return payload


def main() -> None:
    args = parse_args()
    ASRInferencePipeline = _require_omnilingual_pipeline()
    print(
        f"Loading dataset {args.dataset_name}/{args.dataset_config} split={args.split} "
        f"streaming={args.streaming} max_samples={args.max_samples}...",
        flush=True,
    )
    split = load_asr_split(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        materialize_limit=args.max_samples if args.streaming else None,
    )
    if args.max_samples and not args.streaming:
        split = split.select(range(min(args.max_samples, len(split))))

    print(f"Loading Omnilingual model_card={args.model_card} device={args.device or 'auto'}...", flush=True)
    asr_pipeline = ASRInferencePipeline(model_card=args.model_card, device=args.device)
    output_path = args.output_path or f"reports/omnilingual_{args.split}_{args.max_samples or 'full'}_eval.json"
    payload = evaluate_omnilingual_dataset(
        asr_pipeline=asr_pipeline,
        dataset=split,
        lang=args.lang or None,
        batch_size=args.batch_size,
        output_path=output_path,
    )
    summary = {
        "model_card": args.model_card,
        "split": args.split,
        "lang": args.lang or None,
        "output_path": output_path,
        "batch_size": max(int(args.batch_size), 1),
        "metrics": payload["metrics"],
        "sample_count": payload["sample_count"],
    }
    save_json(Path(output_path).with_suffix(".summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
