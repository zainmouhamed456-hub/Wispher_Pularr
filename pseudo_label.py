from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    from datasets import Audio
except ImportError:
    Audio = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from whisper_pularr.data import audio_duration_seconds, load_waxal_asr_dataset, save_json, suggest_num_proc
from whisper_pularr.runtime import runtime_from_optional_report
from whisper_pularr.settings import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_NAME,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_WHISPER_LANGUAGE_HINT,
)
from whisper_pularr.pseudo_label_policy import (
    build_label_profile,
    evaluate_pseudo_label_record,
)
from whisper_pularr.text import compression_ratio


def _require_pseudo_label_runtime() -> None:
    missing: list[str] = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if Audio is None:
        missing.append("datasets[audio]")
    if tqdm is None:
        missing.append("tqdm")
    if missing:
        raise SystemExit(f"Missing pseudo-label dependencies: {', '.join(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for WaxalNLP unlabeled split.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--whisper-language", default=DEFAULT_WHISPER_LANGUAGE_HINT)
    parser.add_argument("--split", default="unlabeled")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hardware-report", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-keep-multiple", type=float, default=2.0)
    parser.add_argument("--candidate-pool-multiple", type=float, default=1.1)
    parser.add_argument("--disable-duration-prioritization", action="store_true")
    parser.add_argument("--min-chars", type=int, default=3)
    parser.add_argument("--avg-logprob-threshold", type=float, default=-0.6)
    parser.add_argument("--compression-ratio-threshold", type=float, default=1.8)
    parser.add_argument("--no-speech-prob-threshold", type=float, default=0.2)
    parser.add_argument("--min-labeled-token-ratio", type=float, default=0.65)
    parser.add_argument("--min-labeled-char-ratio", type=float, default=0.95)
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    parser.add_argument("--manifest-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    return parser.parse_args()


def _teacher_name_for_openai_whisper(model_name: str) -> str:
    if model_name.startswith("openai/whisper-"):
        return model_name.split("openai/whisper-", 1)[1]
    return model_name


def _segment_average(segments: list[dict[str, Any]], key: str, default: float) -> float:
    if not segments:
        return default
    weighted_total = 0.0
    weight_sum = 0.0
    for segment in segments:
        weight = max(float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)), 0.1)
        weighted_total += float(segment.get(key, default)) * weight
        weight_sum += weight
    return weighted_total / max(weight_sum, 1e-6)


def _confidence_score(record: dict[str, Any]) -> float:
    return (
        float(record["avg_logprob"])
        - 0.35 * float(record["compression_ratio"])
        - 0.5 * float(record["no_speech_prob"])
    )


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _shortlist_unlabeled(
    unlabeled: Any,
    *,
    keep_limit: int,
    max_duration_seconds: float,
    candidate_pool_multiple: float,
    prioritize_shorter: bool,
) -> tuple[Any, dict[str, Any]]:
    _require_pseudo_label_runtime()
    undecoded = unlabeled.cast_column("audio", Audio(decode=False))
    num_proc = suggest_num_proc(len(undecoded))
    duration_rows = undecoded.map(
        lambda row: {"duration_seconds": audio_duration_seconds(row)},
        desc="Computing pseudo-label durations",
        num_proc=num_proc,
    )
    candidate_pairs = [
        (index, float(duration))
        for index, duration in enumerate(duration_rows["duration_seconds"])
        if float(duration) <= float(max_duration_seconds)
    ]
    eligible_count = len(candidate_pairs)
    excluded_too_long = len(unlabeled) - eligible_count
    if prioritize_shorter:
        shortlist_size = min(
            eligible_count,
            max(keep_limit, int(math.ceil(keep_limit * max(candidate_pool_multiple, 1.0)))),
        )
        candidate_pairs.sort(key=lambda item: item[1])
        candidate_pairs = candidate_pairs[:shortlist_size]
    selected_indices = [index for index, _ in candidate_pairs]
    selected_durations = [duration for _, duration in candidate_pairs]
    shortlisted = unlabeled.select(selected_indices).add_column("duration_seconds", selected_durations)
    return shortlisted, {
        "eligible_count": eligible_count,
        "excluded_too_long": excluded_too_long,
        "shortlisted_count": len(shortlisted),
        "prioritize_shorter": prioritize_shorter,
        "candidate_pool_multiple": float(candidate_pool_multiple),
    }


def _decode_batch(
    teacher: Any,
    openai_whisper: Any,
    rows: list[dict[str, Any]],
    *,
    device: str,
    language: str | None,
    beam_size: int,
) -> list[Any]:
    _require_pseudo_label_runtime()
    mels = []
    n_mels = int(getattr(getattr(teacher, "dims", None), "n_mels", 80) or 80)
    for row in rows:
        audio = np.asarray(row["audio"]["array"], dtype=np.float32)
        audio = openai_whisper.pad_or_trim(audio)
        mels.append(openai_whisper.log_mel_spectrogram(audio, n_mels=n_mels, device=device))
    mel_batch = torch.stack(mels, dim=0)
    options_kwargs: dict[str, Any] = {
        "task": "transcribe",
        "language": language or None,
        "temperature": 0.0,
        "without_timestamps": True,
        "fp16": bool(device == "cuda"),
    }
    if int(beam_size) > 1:
        options_kwargs["beam_size"] = max(int(beam_size), 1)
    options = openai_whisper.DecodingOptions(**options_kwargs)
    results = openai_whisper.decode(teacher, mel_batch, options)
    if isinstance(results, list):
        return results
    return [results]


def main() -> None:
    args = parse_args()
    _require_pseudo_label_runtime()
    runtime = runtime_from_optional_report(args.hardware_report)
    dataset = load_waxal_asr_dataset(args.dataset_name, args.dataset_config, cache_dir=args.cache_dir or runtime.cache_root)
    label_profile = build_label_profile(dataset["train"])
    unlabeled = dataset[args.split]
    labeled_train_size = len(dataset["train"])
    keep_limit = int(labeled_train_size * args.max_keep_multiple)
    if args.max_samples:
        unlabeled = unlabeled.select(range(min(args.max_samples, len(unlabeled))))
    original_unlabeled_count = len(unlabeled)
    unlabeled, shortlist_stats = _shortlist_unlabeled(
        unlabeled,
        keep_limit=keep_limit,
        max_duration_seconds=args.max_duration_seconds,
        candidate_pool_multiple=args.candidate_pool_multiple,
        prioritize_shorter=not args.disable_duration_prioritization,
    )

    try:
        import whisper as openai_whisper
    except ImportError as exc:
        raise SystemExit("openai-whisper is required for pseudo-label generation.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    teacher_name = _teacher_name_for_openai_whisper(args.teacher_model)
    teacher = openai_whisper.load_model(teacher_name, device=device)
    teacher.eval()
    batch_size = max(int(args.batch_size or runtime.pseudo_label_batch_size), 1)
    beam_size = max(int(args.beam_size or 1), 1)
    whisper_language = str(args.whisper_language or "").strip() or None

    output_path = Path(args.output_path)
    manifest_dir = output_path.parent / f"{output_path.stem}_manifests"

    accepted: list[dict[str, Any]] = []
    rejected = 0
    rejected_low_logprob = 0
    rejected_high_compression = 0
    rejected_high_no_speech = 0
    rejected_language_mismatch = 0
    rejected_too_short = 0
    too_long = int(shortlist_stats["excluded_too_long"])
    processed_count = 0
    manifest_paths: list[str] = []
    progress = tqdm(total=len(unlabeled), desc="Pseudo-labeling")
    with torch.inference_mode():
        for start in range(0, len(unlabeled), batch_size):
            batch = unlabeled[start : start + batch_size]
            batch_rows = [
                {key: values[index] for key, values in batch.items()}
                for index in range(len(batch["id"]))
            ]
            batch_results = _decode_batch(
                teacher,
                openai_whisper,
                batch_rows,
                device=device,
                language=whisper_language,
                beam_size=beam_size,
            )
            for row, result in zip(batch_rows, batch_results):
                processed_count += 1
                duration = float(row.get("duration_seconds") or audio_duration_seconds(row))
                text = (getattr(result, "text", "") or "").strip()
                record = {
                    "id": row["id"],
                    "pseudo_transcription": text,
                    "duration_seconds": duration,
                    "avg_logprob": float(getattr(result, "avg_logprob", -10.0)),
                    "compression_ratio": float(getattr(result, "compression_ratio", compression_ratio(text))),
                    "no_speech_prob": float(getattr(result, "no_speech_prob", 1.0 if not text else 0.0)),
                }
                passes, rejection_reasons = evaluate_pseudo_label_record(
                    record,
                    token_vocab=label_profile["token_vocab"],
                    allowed_chars=label_profile["allowed_chars"],
                    min_chars=args.min_chars,
                    avg_logprob_threshold=args.avg_logprob_threshold,
                    compression_ratio_threshold=args.compression_ratio_threshold,
                    no_speech_prob_threshold=args.no_speech_prob_threshold,
                    min_labeled_token_ratio=args.min_labeled_token_ratio,
                    min_labeled_char_ratio=args.min_labeled_char_ratio,
                )
                if passes:
                    record["confidence_score"] = _confidence_score(record)
                    accepted.append(record)
                else:
                    rejected += 1
                    if "low_logprob" in rejection_reasons:
                        rejected_low_logprob += 1
                    if "high_compression" in rejection_reasons:
                        rejected_high_compression += 1
                    if "high_no_speech" in rejection_reasons:
                        rejected_high_no_speech += 1
                    if "language_mismatch" in rejection_reasons:
                        rejected_language_mismatch += 1
                    if "too_short" in rejection_reasons:
                        rejected_too_short += 1

                if args.manifest_every > 0 and processed_count % args.manifest_every == 0:
                    snapshot = sorted(accepted, key=lambda item: item["confidence_score"], reverse=True)[:keep_limit]
                    manifest_path = manifest_dir / f"pseudo_labels_{processed_count:06d}.jsonl"
                    _write_manifest(manifest_path, snapshot)
                    manifest_paths.append(str(manifest_path))
            progress.update(len(batch_rows))
    progress.close()

    accepted.sort(key=lambda row: row["confidence_score"], reverse=True)
    accepted = accepted[:keep_limit]

    _write_manifest(output_path, accepted)
    final_snapshot_path = manifest_dir / "pseudo_labels_final.jsonl"
    _write_manifest(final_snapshot_path, accepted)
    manifest_paths.append(str(final_snapshot_path))

    report = {
        "teacher_model": args.teacher_model,
        "whisper_language": whisper_language,
        "input_split": args.split,
        "input_samples": original_unlabeled_count,
        "duration_eligible_samples": shortlist_stats["eligible_count"],
        "shortlisted_samples": shortlist_stats["shortlisted_count"],
        "accepted_samples": len(accepted),
        "rejected_samples": rejected,
        "rejected_low_logprob": rejected_low_logprob,
        "rejected_high_compression": rejected_high_compression,
        "rejected_high_no_speech": rejected_high_no_speech,
        "rejected_language_mismatch": rejected_language_mismatch,
        "rejected_too_short": rejected_too_short,
        "too_long_samples": too_long,
        "keep_limit": keep_limit,
        "acceptance_rate": float(len(accepted) / max(len(unlabeled), 1)),
        "duration_prioritization": not args.disable_duration_prioritization,
        "candidate_pool_multiple": float(args.candidate_pool_multiple),
        "manifest_every": int(args.manifest_every),
        "batch_size": int(batch_size),
        "beam_size": int(beam_size),
        "min_labeled_token_ratio": float(args.min_labeled_token_ratio),
        "min_labeled_char_ratio": float(args.min_labeled_char_ratio),
        "processed_samples": processed_count,
        "manifest_dir": str(manifest_dir),
        "manifest_paths": manifest_paths,
        "output_path": str(output_path),
        "runtime": runtime.to_dict(),
    }
    report_path = args.report_path or str(output_path.with_suffix(".report.json"))
    save_json(report_path, report)
    print(report)


if __name__ == "__main__":
    main()
