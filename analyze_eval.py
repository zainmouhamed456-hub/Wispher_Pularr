from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from whisper_pularr.data import save_json
from whisper_pularr.eval_utils import compute_error_metrics
from whisper_pularr.text import normalize_transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize error patterns from a saved evaluation JSON payload.")
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def _boundary_fold(text: str) -> str:
    return normalize_transcript(text).replace(" ", "").replace("-", "").replace("'", "")


def _mark_fold(text: str) -> str:
    return normalize_transcript(text).replace("-", "").replace("'", "")


def _categorize_sample(reference: str, prediction: str) -> str:
    normalized_reference = normalize_transcript(reference)
    normalized_prediction = normalize_transcript(prediction)
    if reference == prediction:
        return "raw_exact_match"
    if normalized_reference == normalized_prediction:
        return "normalization_only_match"
    if _boundary_fold(reference) == _boundary_fold(prediction):
        return "word_boundary_or_mark_mismatch"
    if _mark_fold(reference) == _mark_fold(prediction):
        return "apostrophe_or_hyphen_mismatch"
    return "content_mismatch"


def _sample_record(sample: dict[str, Any]) -> dict[str, Any]:
    reference = str(sample.get("reference") or "")
    prediction = str(sample.get("prediction") or "")
    metrics = compute_error_metrics([reference], [prediction])
    return {
        "id": sample.get("id"),
        "reference": reference,
        "prediction": prediction,
        "normalized_reference": normalize_transcript(reference),
        "normalized_prediction": normalize_transcript(prediction),
        "category": _categorize_sample(reference, prediction),
        "raw_wer": metrics["raw_wer"],
        "raw_cer": metrics["raw_cer"],
        "normalized_wer": metrics["normalized_wer"],
        "normalized_cer": metrics["normalized_cer"],
    }


def _top_examples(records: list[dict[str, Any]], *, category: str, limit: int) -> list[dict[str, Any]]:
    selected = [record for record in records if record["category"] == category]
    selected.sort(
        key=lambda record: (
            -float(record["normalized_wer"]),
            -float(record["normalized_cer"]),
            str(record.get("id") or ""),
        )
    )
    return selected[:limit]


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_json)
    with eval_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = [_sample_record(sample) for sample in payload.get("samples", [])]
    mismatch_counter = Counter(
        (record["normalized_reference"], record["normalized_prediction"])
        for record in records
        if record["normalized_reference"] != record["normalized_prediction"]
    )
    category_counter = Counter(record["category"] for record in records)
    worst_samples = sorted(
        records,
        key=lambda record: (
            -float(record["normalized_wer"]),
            -float(record["normalized_cer"]),
            -float(record["raw_wer"]),
            -float(record["raw_cer"]),
        ),
    )[: max(int(args.top_k), 1)]

    summary = {
        "eval_json": str(eval_path),
        "sample_count": len(records),
        "metrics": payload.get("metrics") or {},
        "category_counts": dict(sorted(category_counter.items())),
        "top_mismatch_pairs": [
            {
                "normalized_reference": reference,
                "normalized_prediction": prediction,
                "count": count,
            }
            for (reference, prediction), count in mismatch_counter.most_common(20)
        ],
        "worst_samples": worst_samples,
        "likely_word_boundary_examples": _top_examples(
            records,
            category="word_boundary_or_mark_mismatch",
            limit=min(max(int(args.top_k), 1), 25),
        ),
        "likely_apostrophe_hyphen_examples": _top_examples(
            records,
            category="apostrophe_or_hyphen_mismatch",
            limit=min(max(int(args.top_k), 1), 25),
        ),
    }

    output_path = args.output_path or str(eval_path.with_suffix(".analysis.json"))
    save_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
