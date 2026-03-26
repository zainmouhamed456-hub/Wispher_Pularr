from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
except ImportError:
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None

try:
    import torch
except ImportError:
    torch = None

from whisper_pularr.data import load_waxal_asr_dataset, save_json
from whisper_pularr.eval_utils import evaluate_long_form_dataset
from whisper_pularr.runtime import runtime_from_optional_report
from whisper_pularr.settings import DEFAULT_DATASET_CONFIG, DEFAULT_DATASET_NAME, DEFAULT_WHISPER_LANGUAGE_HINT
from whisper_pularr.whisper_prompt import configure_whisper_prompt, resolve_whisper_language


def _require_compare_runtime() -> None:
    if AutoModelForSpeechSeq2Seq is None or AutoProcessor is None:
        raise SystemExit("Missing comparison dependencies: transformers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple Whisper checkpoints on a fixed slice and full split.")
    parser.add_argument("--checkpoint", action="append", dest="checkpoints", required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", choices=["validation", "test", "train"], default="validation")
    parser.add_argument("--fixed-slice-size", type=int, default=32)
    parser.add_argument("--whisper-language", default=DEFAULT_WHISPER_LANGUAGE_HINT)
    parser.add_argument("--hardware-report", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-full", action="store_true")
    return parser.parse_args()


def _safe_name(checkpoint: str) -> str:
    return (
        checkpoint.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "")
        .replace(".", "_")
    )


def _evaluate_checkpoint(
    checkpoint: str,
    *,
    dataset: Any,
    fixed_slice: Any,
    runtime: Any,
    output_dir: Path,
    whisper_language: str | None,
    skip_full: bool,
) -> dict[str, Any]:
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(checkpoint)
    language = resolve_whisper_language(processor.tokenizer, whisper_language)
    configure_whisper_prompt(processor=processor, model=model, language=language)

    checkpoint_dir = output_dir / _safe_name(checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fixed_payload = evaluate_long_form_dataset(
        model=model,
        processor=processor,
        dataset=fixed_slice,
        runtime_config=runtime,
        output_path=str(checkpoint_dir / "fixed_slice_eval.json"),
        language=language,
    )

    full_payload = None
    if not skip_full:
        full_payload = evaluate_long_form_dataset(
            model=model,
            processor=processor,
            dataset=dataset,
            runtime_config=runtime,
            output_path=str(checkpoint_dir / "full_eval.json"),
            language=language,
        )

    summary = {
        "checkpoint": checkpoint,
        "resolved_language": language,
        "fixed_slice_size": fixed_payload["sample_count"],
        "fixed_slice_metrics": fixed_payload["metrics"],
        "full_metrics": full_payload["metrics"] if full_payload else None,
        "fixed_slice_output_path": str(checkpoint_dir / "fixed_slice_eval.json"),
        "full_output_path": None if skip_full else str(checkpoint_dir / "full_eval.json"),
    }
    save_json(checkpoint_dir / "summary.json", summary)
    del model
    del processor
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    _require_compare_runtime()
    runtime = runtime_from_optional_report(args.hardware_report)
    cache_dir = args.cache_dir or runtime.cache_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_waxal_asr_dataset(args.dataset_name, args.dataset_config, cache_dir=cache_dir)[args.split]
    fixed_slice = dataset.select(range(min(len(dataset), max(int(args.fixed_slice_size), 1))))

    summaries = [
        _evaluate_checkpoint(
            checkpoint,
            dataset=dataset,
            fixed_slice=fixed_slice,
            runtime=runtime,
            output_dir=output_dir,
            whisper_language=args.whisper_language,
            skip_full=bool(args.skip_full),
        )
        for checkpoint in args.checkpoints
    ]
    save_json(
        output_dir / "comparison_summary.json",
        {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "fixed_slice_size": len(fixed_slice),
            "skip_full": bool(args.skip_full),
            "checkpoints": summaries,
        },
    )


if __name__ == "__main__":
    main()
