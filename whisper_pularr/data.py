from __future__ import annotations

import json
import math
import os
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
except ImportError:
    Audio = None
    Dataset = Any
    DatasetDict = Any
    concatenate_datasets = None
    load_dataset = None

from .settings import (
    DEFAULT_AUDIO_SAMPLING_RATE,
    DEFAULT_MAX_TRAIN_DURATION_SECONDS,
    DEFAULT_TRAINING_LANGUAGE_CANDIDATES,
)
from .text import normalize_transcript
from .whisper_prompt import configure_whisper_prompt as apply_whisper_prompt


def _require_datasets() -> None:
    if Audio is None or concatenate_datasets is None or load_dataset is None:
        raise RuntimeError("datasets[audio] is required for dataset operations.")


def _load_dataset_kwargs(dataset_name: str, cache_dir: str | None = None) -> dict[str, Any]:
    return {
        "cache_dir": cache_dir,
        "trust_remote_code": dataset_name == "google/fleurs",
    }


def load_asr_split(
    dataset_name: str,
    dataset_config: str,
    split: str,
    cache_dir: str | None = None,
    streaming: bool = False,
    materialize_limit: int | None = None,
) -> Dataset:
    _require_datasets()
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=streaming,
        **_load_dataset_kwargs(dataset_name, cache_dir=cache_dir),
    )
    if streaming:
        stream = dataset if materialize_limit is None else islice(dataset, max(int(materialize_limit), 0))
        rows = [dict(row) for row in stream]
        materialized = Dataset.from_list(rows)
        return materialized.cast_column("audio", Audio(sampling_rate=DEFAULT_AUDIO_SAMPLING_RATE))
    return dataset.cast_column("audio", Audio(sampling_rate=DEFAULT_AUDIO_SAMPLING_RATE))


def load_waxal_asr_dataset(
    dataset_name: str,
    dataset_config: str,
    cache_dir: str | None = None,
) -> DatasetDict:
    _require_datasets()
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        **_load_dataset_kwargs(dataset_name, cache_dir=cache_dir),
    )
    return dataset.cast_column("audio", Audio(sampling_rate=DEFAULT_AUDIO_SAMPLING_RATE))


def audio_duration_seconds(example: dict[str, Any]) -> float:
    audio = example["audio"]
    if audio is None:
        return 0.0
    array = audio.get("array")
    if array is not None:
        sample_rate = int(audio.get("sampling_rate") or audio.get("sample_rate") or DEFAULT_AUDIO_SAMPLING_RATE)
        return float(len(array) / max(sample_rate, 1))
    duration = audio.get("duration")
    if duration is not None:
        return float(duration)
    if torchaudio is not None:
        for source in (
            BytesIO(audio["bytes"]) if audio.get("bytes") is not None else None,
            audio.get("path"),
        ):
            if source is None:
                continue
            try:
                info = torchaudio.info(source)
            except Exception:
                continue
            if info.sample_rate and info.num_frames:
                return float(info.num_frames / info.sample_rate)
    if audio.get("bytes") is not None or audio.get("path") is not None:
        return float("inf")
    return 0.0


def with_duration(example: dict[str, Any]) -> dict[str, Any]:
    example["duration_seconds"] = audio_duration_seconds(example)
    return example


def suggest_num_proc(dataset_length: int, max_workers: int = 24) -> int | None:
    if os.name == "nt":
        return None
    if Path("/content").exists() or os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return None
    cpu_total = os.cpu_count() or 1
    suggested = min(max(cpu_total - 1, 1), max_workers, max(int(dataset_length), 1))
    return suggested if suggested > 1 else None


def filter_by_max_duration(dataset: Dataset, max_duration_seconds: float = DEFAULT_MAX_TRAIN_DURATION_SECONDS) -> Dataset:
    _require_datasets()
    if "duration_seconds" not in dataset.column_names:
        undecoded = dataset.cast_column("audio", Audio(decode=False))
        num_proc = suggest_num_proc(len(undecoded))
        durations = undecoded.map(
            lambda row: {"duration_seconds": audio_duration_seconds(row)},
            desc="Computing audio durations",
            num_proc=num_proc,
        )["duration_seconds"]
        dataset = dataset.add_column("duration_seconds", durations)
    return dataset.filter(
        lambda row: math.isfinite(float(row["duration_seconds"]))
        and float(row["duration_seconds"]) <= float(max_duration_seconds)
    )


def duplicate_dataset(dataset: Dataset, repeats: int) -> Dataset:
    _require_datasets()
    if repeats <= 1:
        return dataset
    return concatenate_datasets([dataset] * repeats)


def load_pseudo_label_manifest(path: str) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            manifest[str(record["id"])] = record
    return manifest


def attach_pseudo_labels(unlabeled: Dataset, manifest_path: str) -> Dataset:
    manifest = load_pseudo_label_manifest(manifest_path)
    if not manifest:
        return unlabeled.select([])

    index_by_id = {str(row_id): index for index, row_id in enumerate(unlabeled["id"])}
    selected_indices = [index_by_id[row_id] for row_id in manifest if row_id in index_by_id]
    selected = unlabeled.select(selected_indices)

    def _attach_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        transcriptions: list[str] = []
        avg_logprobs: list[float | None] = []
        compression_ratios: list[float | None] = []
        no_speech_probs: list[float | None] = []
        durations: list[float | None] = []
        sources: list[str | None] = []
        for row_id in batch["id"]:
            record = manifest.get(str(row_id))
            transcriptions.append(record["pseudo_transcription"] if record else "")
            avg_logprobs.append(record["avg_logprob"] if record else None)
            compression_ratios.append(record["compression_ratio"] if record else None)
            no_speech_probs.append(record["no_speech_prob"] if record else None)
            durations.append(record["duration_seconds"] if record else None)
            sources.append("teacher" if record else None)
        batch["transcription"] = transcriptions
        batch["pseudo_avg_logprob"] = avg_logprobs
        batch["pseudo_compression_ratio"] = compression_ratios
        batch["pseudo_no_speech_prob"] = no_speech_probs
        batch["duration_seconds"] = durations
        batch["pseudo_source"] = sources
        return batch

    return selected.map(_attach_batch, batched=True)


def build_stage_dataset(
    dataset: DatasetDict,
    stage: str,
    pseudo_labels_path: str | None = None,
    labeled_repeat_count: int = 2,
) -> Dataset:
    _require_datasets()
    train = filter_by_max_duration(dataset["train"])
    if stage == "supervised":
        return train
    if stage != "self_train":
        raise ValueError(f"Unsupported stage: {stage}")
    if not pseudo_labels_path:
        raise ValueError("Self-training requires --pseudo-labels-path.")

    pseudo = attach_pseudo_labels(dataset["unlabeled"], pseudo_labels_path)
    pseudo = filter_by_max_duration(pseudo)
    labeled = duplicate_dataset(train, labeled_repeat_count)
    return concatenate_datasets([labeled, pseudo]).shuffle(seed=42)


def infer_whisper_language(tokenizer: Any, candidates: Iterable[str] = DEFAULT_TRAINING_LANGUAGE_CANDIDATES) -> str | None:
    lang_to_id = getattr(tokenizer, "lang_to_id", None)
    if not isinstance(lang_to_id, dict):
        return None
    lowered_map = {key.lower(): key for key in lang_to_id}
    for candidate in candidates:
        actual = lowered_map.get(candidate.lower())
        if actual:
            return actual
    return None


def configure_whisper_prompt(processor: Any, model: Any | None = None, language: str | None = None) -> list[tuple[int, int]] | None:
    apply_whisper_prompt(processor=processor, model=model, language=language)
    return None


def prepare_training_example(
    example: dict[str, Any],
    processor: Any,
    max_label_length: int | None = None,
) -> dict[str, Any]:
    audio = example["audio"]
    features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )
    labels = processor.tokenizer(normalize_transcript(example["transcription"])).input_ids
    if max_label_length:
        labels = labels[:max_label_length]
    return {
        "input_features": features["input_features"][0],
        "labels": labels,
        "input_length": len(audio["array"]),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
