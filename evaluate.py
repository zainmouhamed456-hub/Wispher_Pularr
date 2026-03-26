from __future__ import annotations

import argparse
from pathlib import Path

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
except ImportError:
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None

from whisper_pularr.data import load_waxal_asr_dataset, save_json
from whisper_pularr.eval_utils import evaluate_long_form_dataset
from whisper_pularr.runtime import runtime_from_optional_report
from whisper_pularr.settings import DEFAULT_DATASET_CONFIG, DEFAULT_DATASET_NAME, DEFAULT_WHISPER_LANGUAGE_HINT
from whisper_pularr.whisper_prompt import configure_whisper_prompt, resolve_whisper_language


def _require_eval_runtime() -> None:
    if AutoModelForSpeechSeq2Seq is None or AutoProcessor is None:
        raise SystemExit("Missing evaluation dependencies: transformers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Whisper checkpoint on WaxalNLP.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", choices=["validation", "test", "train"], default="validation")
    parser.add_argument("--whisper-language", default=DEFAULT_WHISPER_LANGUAGE_HINT)
    parser.add_argument("--hardware-report", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_eval_runtime()
    runtime = runtime_from_optional_report(args.hardware_report)
    dataset = load_waxal_asr_dataset(args.dataset_name, args.dataset_config, cache_dir=args.cache_dir or runtime.cache_root)
    split = dataset[args.split]
    if args.max_samples:
        split = split.select(range(min(args.max_samples, len(split))))

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.checkpoint)
    language = resolve_whisper_language(processor.tokenizer, args.whisper_language)
    configure_whisper_prompt(processor=processor, model=model, language=language)

    output_path = args.output_path or str(Path(args.checkpoint) / f"{args.split}_long_form_eval.json")
    payload = evaluate_long_form_dataset(
        model=model,
        processor=processor,
        dataset=split,
        runtime_config=runtime,
        output_path=output_path,
        language=language,
    )
    summary = {
        "split": args.split,
        "requested_whisper_language": args.whisper_language,
        "resolved_whisper_language": language,
        "output_path": output_path,
        "metrics": payload["metrics"],
        "sample_count": payload["sample_count"],
    }
    save_json(Path(output_path).with_suffix(".summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
