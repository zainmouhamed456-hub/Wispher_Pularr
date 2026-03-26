from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import jiwer
except ImportError:
    jiwer = None

try:
    import torch
except ImportError:
    torch = None

try:
    from datasets import Dataset, concatenate_datasets
except ImportError:
    Dataset = Any
    concatenate_datasets = None

try:
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
except ImportError:
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    Seq2SeqTrainer = None
    Seq2SeqTrainingArguments = None

from whisper_pularr.data import (
    attach_pseudo_labels,
    build_stage_dataset,
    configure_whisper_prompt,
    duplicate_dataset,
    filter_by_max_duration,
    load_asr_split,
    load_waxal_asr_dataset,
    save_json,
    suggest_num_proc,
)
from whisper_pularr.eval_utils import LongFormEvalAndStopCallback
from whisper_pularr.runtime import runtime_from_optional_report
from whisper_pularr.settings import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_NAME,
    DEFAULT_EARLY_STOP_PATIENCE_EPOCHS,
    DEFAULT_MAX_TRAIN_DURATION_SECONDS,
    DEFAULT_TRAINABLE_MODEL,
    DEFAULT_WHISPER_LANGUAGE_HINT,
    SUPERVISED_PRESETS,
)
from whisper_pularr.text import normalize_transcript
from whisper_pularr.training_policy import resolve_label_smoothing_factor, runtime_for_stage
from whisper_pularr.whisper_prompt import resolve_whisper_language


def _require_train_runtime() -> None:
    missing: list[str] = []
    if jiwer is None:
        missing.append("jiwer")
    if torch is None:
        missing.append("torch")
    if concatenate_datasets is None:
        missing.append("datasets[audio]")
    if (
        AutoModelForSpeechSeq2Seq is None
        or AutoProcessor is None
        or Seq2SeqTrainer is None
        or Seq2SeqTrainingArguments is None
    ):
        missing.append("transformers")
    if missing:
        raise SystemExit(f"Missing training dependencies: {', '.join(missing)}")


@dataclass
class WhisperDataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    model: Any | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            batch["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Whisper Small on WaxalNLP Pularr/Pulaar.")
    parser.add_argument("--stage", choices=["supervised", "self_train"], required=True)
    parser.add_argument("--preset", default="trial_a", choices=sorted(SUPERVISED_PRESETS))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--aux-dataset-name", default=None)
    parser.add_argument("--aux-dataset-config", default=None)
    parser.add_argument("--aux-labeled-repeat-count", type=int, default=1)
    parser.add_argument("--model-id", default=DEFAULT_TRAINABLE_MODEL)
    parser.add_argument("--whisper-language", default=DEFAULT_WHISPER_LANGUAGE_HINT)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--pseudo-labels-path", default=None)
    parser.add_argument("--hardware-report", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-label-length", type=int, default=256)
    parser.add_argument("--max-train-duration-seconds", type=float, default=DEFAULT_MAX_TRAIN_DURATION_SECONDS)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeled-repeat-count", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--early-stop-patience-epochs", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def select_num_epochs(stage: str, explicit_epochs: float | None) -> float:
    if explicit_epochs is not None:
        return explicit_epochs
    return 25.0 if stage == "supervised" else 8.0


def resolve_labeled_repeat_count(stage: str, explicit_repeats: int | None) -> int:
    if explicit_repeats is not None:
        return max(int(explicit_repeats), 1)
    return 2 if stage == "supervised" else 1


def resolve_early_stop_patience(stage: str, explicit_patience: int | None) -> int:
    if explicit_patience is not None:
        return max(int(explicit_patience), 0)
    return DEFAULT_EARLY_STOP_PATIENCE_EPOCHS if stage == "supervised" else 1


def resolve_learning_rate(preset: str, explicit_learning_rate: float | None) -> float:
    if explicit_learning_rate is not None:
        return float(explicit_learning_rate)
    return float(SUPERVISED_PRESETS[preset]["learning_rate"])


def normalize_supervised_split_schema(split: Dataset) -> Dataset:
    normalized = split
    rename_candidates = {
        "transcription": ("text", "sentence", "transcript"),
        "audio": ("speech",),
    }
    for target, candidates in rename_candidates.items():
        if target in normalized.column_names:
            continue
        for candidate in candidates:
            if candidate in normalized.column_names:
                normalized = normalized.rename_column(candidate, target)
                break

    keep_columns = [column for column in ("id", "audio", "transcription", "duration_seconds") if column in normalized.column_names]
    normalized = normalized.select_columns(keep_columns)
    if "id" in normalized.column_names:
        return normalized.map(
            lambda batch: {"id": [str(value) for value in batch["id"]]},
            batched=True,
            desc="Normalizing supervised ids",
        )
    return normalized.map(
        lambda _batch, indices: {"id": [str(index) for index in indices]},
        batched=True,
        with_indices=True,
        desc="Synthesizing supervised ids",
    )


def _split_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def preprocess_split(
    split: Dataset,
    processor: Any,
    max_label_length: int,
    max_samples: int | None = None,
    batch_size: int = 32,
    num_proc: int | None = None,
) -> Dataset:
    if max_samples:
        split = split.select(range(min(max_samples, len(split))))
    remove_columns = split.column_names
    if num_proc is None:
        num_proc = suggest_num_proc(len(split))
    return split.map(
        lambda batch: _prepare_training_batch(batch, processor=processor, max_label_length=max_label_length),
        remove_columns=remove_columns,
        desc="Preparing training features",
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )


def _prepare_training_batch(
    batch: dict[str, list[Any]],
    *,
    processor: Any,
    max_label_length: int,
) -> dict[str, list[Any]]:
    audios = batch["audio"]
    arrays = [audio["array"] for audio in audios]
    sampling_rate = audios[0]["sampling_rate"] if audios else 16_000
    features = processor.feature_extractor(
        arrays,
        sampling_rate=sampling_rate,
        return_attention_mask=False,
    )
    labels = processor.tokenizer(
        [normalize_transcript(text) for text in batch["transcription"]],
    ).input_ids
    if max_label_length:
        labels = [label[:max_label_length] for label in labels]
    return {
        "input_features": features["input_features"],
        "labels": labels,
        "input_length": [len(audio["array"]) for audio in audios],
    }


def build_compute_metrics(processor: Any):
    _require_train_runtime()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        norm_preds = [normalize_transcript(text) for text in pred_str]
        norm_labels = [normalize_transcript(text) for text in label_str]
        return {
            "wer": float(jiwer.wer(norm_labels, norm_preds)),
            "cer": float(jiwer.cer(norm_labels, norm_preds)),
        }

    return compute_metrics


def prepare_model_and_processor(model_name_or_path: str) -> tuple[Any, Any]:
    _require_train_runtime()
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path)
    return model, processor


def resolve_base_checkpoint(stage: str, explicit_base_checkpoint: str | None) -> str | None:
    if explicit_base_checkpoint:
        return explicit_base_checkpoint
    if stage != "self_train":
        return None

    downloads_root = Path(__file__).resolve().parent / "downloads"
    if not downloads_root.exists():
        return None

    preferred_checkpoint = downloads_root / "trial_a_best_full_eval" / "best_full_eval"
    if preferred_checkpoint.is_dir():
        return str(preferred_checkpoint)

    candidates: list[tuple[tuple[float, float], Path]] = []
    for summary_path in downloads_root.glob("*/run_summary.json"):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        metrics = summary.get("best_metrics") or {}
        checkpoint_path = summary_path.parent / "best_full_eval"
        if not checkpoint_path.is_dir():
            continue
        wer = float(metrics.get("normalized_wer", float("inf")))
        cer = float(metrics.get("normalized_cer", float("inf")))
        candidates.append(((wer, cer), checkpoint_path))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return str(candidates[0][1])

    fallback_paths = sorted(path for path in downloads_root.glob("*/best_full_eval") if path.is_dir())
    if fallback_paths:
        return str(fallback_paths[0])
    return None


def load_auxiliary_train_split(
    *,
    aux_dataset_name: str | None,
    aux_dataset_config: str | None,
    cache_dir: str | None,
    max_train_duration_seconds: float,
    aux_labeled_repeat_count: int,
    reference_train_size: int,
    seed: int,
) -> Dataset | None:
    if not aux_dataset_name:
        return None
    if not aux_dataset_config:
        raise ValueError("--aux-dataset-config is required when --aux-dataset-name is provided.")

    dataset_names = _split_csv_arg(aux_dataset_name)
    dataset_configs = _split_csv_arg(aux_dataset_config)
    if len(dataset_names) != len(dataset_configs):
        raise ValueError("Auxiliary dataset names and configs must have the same comma-separated length.")

    per_dataset_cap = max(int(reference_train_size), 1) * max(int(aux_labeled_repeat_count), 1)
    auxiliary_splits: list[Dataset] = []
    for dataset_name, dataset_config in zip(dataset_names, dataset_configs):
        auxiliary_train = load_asr_split(
            dataset_name,
            dataset_config,
            split="train",
            cache_dir=cache_dir,
        )
        auxiliary_train = filter_by_max_duration(auxiliary_train, max_train_duration_seconds)
        auxiliary_train = normalize_supervised_split_schema(auxiliary_train)
        auxiliary_train = duplicate_dataset(auxiliary_train, max(int(aux_labeled_repeat_count), 1))
        if len(auxiliary_train) > per_dataset_cap:
            auxiliary_train = auxiliary_train.shuffle(seed=seed).select(range(per_dataset_cap))
        auxiliary_splits.append(auxiliary_train)

    if not auxiliary_splits:
        return None
    return concatenate_datasets(auxiliary_splits).shuffle(seed=seed)


def main() -> None:
    args = parse_args()
    _require_train_runtime()
    runtime = runtime_from_optional_report(args.hardware_report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    runtime = runtime_for_stage(args.stage, runtime, use_cuda)
    use_streaming = bool(args.streaming or (args.stage == "supervised" and getattr(runtime, "profile", None) == "colab_t4"))
    preprocess_batch_size = 4 if getattr(runtime, "profile", None) == "colab_t4" else 32
    preprocess_num_proc = None if getattr(runtime, "profile", None) != "colab_t4" else 1
    use_trainer_eval = not (args.stage == "supervised" and getattr(runtime, "profile", None) == "colab_t4")

    cache_dir = args.cache_dir or runtime.cache_root
    dataset = (
        load_waxal_asr_dataset(args.dataset_name, args.dataset_config, cache_dir=cache_dir)
        if args.stage == "self_train"
        else {
            "train": load_asr_split(
                args.dataset_name,
                args.dataset_config,
                split="train",
                cache_dir=cache_dir,
                streaming=use_streaming,
                materialize_limit=args.max_train_samples,
            ),
            "validation": load_asr_split(
                args.dataset_name,
                args.dataset_config,
                split="validation",
                cache_dir=cache_dir,
                streaming=use_streaming,
                materialize_limit=args.max_eval_samples,
            ),
        }
    )

    resolved_base_checkpoint = resolve_base_checkpoint(args.stage, args.base_checkpoint)
    model_source = resolved_base_checkpoint or args.model_id
    model, processor = prepare_model_and_processor(model_source)
    language = resolve_whisper_language(processor.tokenizer, args.whisper_language)
    configure_whisper_prompt(processor=processor, model=model, language=language)
    model.generation_config.num_beams = runtime.generation_num_beams
    labeled_repeat_count = resolve_labeled_repeat_count(args.stage, args.labeled_repeat_count)
    early_stop_patience_epochs = resolve_early_stop_patience(args.stage, args.early_stop_patience_epochs)
    label_smoothing_factor = resolve_label_smoothing_factor(args.preset)

    model.config.use_cache = False
    if hasattr(model.config, "apply_spec_augment"):
        model.config.apply_spec_augment = bool(SUPERVISED_PRESETS[args.preset]["apply_spec_augment"])

    if args.stage == "self_train":
        labeled_train = filter_by_max_duration(dataset["train"], args.max_train_duration_seconds)
        pseudo_split = attach_pseudo_labels(dataset["unlabeled"], args.pseudo_labels_path or "")
        pseudo_split = filter_by_max_duration(pseudo_split, args.max_train_duration_seconds)
        train_split = concatenate_datasets([duplicate_dataset(labeled_train, labeled_repeat_count), pseudo_split]).shuffle(seed=args.seed)
        reference_train_size = len(labeled_train)
        processed_labeled = preprocess_split(
            labeled_train,
            processor,
            args.max_label_length,
            batch_size=preprocess_batch_size,
            num_proc=preprocess_num_proc,
        )
        processed_pseudo = preprocess_split(
            pseudo_split,
            processor,
            args.max_label_length,
            batch_size=preprocess_batch_size,
            num_proc=preprocess_num_proc,
        )
        processed_train = concatenate_datasets(
            [duplicate_dataset(processed_labeled, labeled_repeat_count), processed_pseudo]
        ).shuffle(seed=args.seed)
        validation_for_trainer = None
    else:
        train_split = build_stage_dataset(
            dataset=dataset,
            stage=args.stage,
            pseudo_labels_path=args.pseudo_labels_path,
            labeled_repeat_count=labeled_repeat_count,
        )
        train_split = normalize_supervised_split_schema(train_split)
        reference_train_size = len(train_split)
        auxiliary_train = load_auxiliary_train_split(
            aux_dataset_name=args.aux_dataset_name,
            aux_dataset_config=args.aux_dataset_config,
            cache_dir=cache_dir,
            max_train_duration_seconds=args.max_train_duration_seconds,
            aux_labeled_repeat_count=args.aux_labeled_repeat_count,
            reference_train_size=reference_train_size,
            seed=args.seed,
        )
        if auxiliary_train is not None:
            train_split = concatenate_datasets([train_split, auxiliary_train]).shuffle(seed=args.seed)
        processed_train = preprocess_split(
            train_split,
            processor,
            args.max_label_length,
            batch_size=preprocess_batch_size,
            num_proc=preprocess_num_proc,
        )
        validation_for_trainer = filter_by_max_duration(dataset["validation"], args.max_train_duration_seconds)
    validation_for_full_eval = dataset["validation"]

    if args.max_train_samples:
        train_split = train_split.select(range(min(args.max_train_samples, len(train_split))))
        processed_train = processed_train.select(range(min(args.max_train_samples, len(processed_train))))
    if args.max_eval_samples:
        if validation_for_trainer is not None:
            validation_for_trainer = validation_for_trainer.select(range(min(args.max_eval_samples, len(validation_for_trainer))))
        validation_for_full_eval = validation_for_full_eval.select(range(min(args.max_eval_samples, len(validation_for_full_eval))))

    processed_validation = None
    if validation_for_trainer is not None and use_trainer_eval:
        processed_validation = preprocess_split(
            validation_for_trainer,
            processor,
            args.max_label_length,
            batch_size=preprocess_batch_size,
            num_proc=preprocess_num_proc,
        )

    data_collator = WhisperDataCollatorSpeechSeq2SeqWithPadding(processor=processor, model=model)
    evaluation_strategy = "epoch" if args.stage == "supervised" and use_trainer_eval else "no"
    training_args_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=runtime.per_device_train_batch_size,
        per_device_eval_batch_size=runtime.per_device_eval_batch_size,
        gradient_accumulation_steps=runtime.gradient_accumulation_steps,
        learning_rate=resolve_learning_rate(args.preset, args.learning_rate),
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        weight_decay=args.weight_decay,
        num_train_epochs=select_num_epochs(args.stage, args.num_train_epochs),
        evaluation_strategy=evaluation_strategy,
        predict_with_generate=bool(args.stage == "supervised"),
        generation_max_length=args.max_label_length,
        generation_num_beams=runtime.generation_num_beams,
        save_strategy="steps" if args.stage == "supervised" else "no",
        save_steps=runtime.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=runtime.logging_steps,
        dataloader_num_workers=runtime.dataloader_num_workers,
        dataloader_pin_memory=use_cuda,
        bf16=runtime.bf16 if use_cuda else False,
        fp16=runtime.fp16 if use_cuda else False,
        gradient_checkpointing=bool(args.stage == "supervised"),
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        label_smoothing_factor=label_smoothing_factor,
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=False,
    )
    if int(runtime.dataloader_num_workers or 0) > 0:
        training_args_kwargs["dataloader_prefetch_factor"] = runtime.dataloader_prefetch_factor
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    callback = LongFormEvalAndStopCallback(
        processor=processor,
        eval_dataset=validation_for_full_eval,
        runtime_config=runtime,
        output_dir=str(output_dir),
        language=language,
        early_stop_patience_epochs=early_stop_patience_epochs,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_validation,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=build_compute_metrics(processor) if processed_validation is not None and use_trainer_eval else None,
        callbacks=[callback],
    )

    train_result = trainer.train()

    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    processor.save_pretrained(final_model_dir)

    train_metrics = train_result.metrics
    trainer_metrics_path = output_dir / "trainer_metrics.json"
    save_json(trainer_metrics_path, train_metrics)

    effective_config = {
        "args": vars(args),
        "runtime": runtime.to_dict(),
        "requested_whisper_language": args.whisper_language,
        "language": language,
        "forced_decoder_ids": None,
        "requested_label_smoothing_factor": label_smoothing_factor,
        "applied_label_smoothing_factor": label_smoothing_factor,
        "requested_learning_rate": resolve_learning_rate(args.preset, args.learning_rate),
        "train_samples": len(train_split),
        "labeled_repeat_count": labeled_repeat_count,
        "aux_dataset_name": _split_csv_arg(args.aux_dataset_name),
        "aux_dataset_config": _split_csv_arg(args.aux_dataset_config),
        "aux_labeled_repeat_count": args.aux_labeled_repeat_count,
        "aux_sampling_cap_per_dataset": max(int(reference_train_size), 1) * max(int(args.aux_labeled_repeat_count), 1),
        "early_stop_patience_epochs": early_stop_patience_epochs,
        "validation_samples_trainer": len(validation_for_trainer) if validation_for_trainer is not None else 0,
        "validation_samples_full_eval": len(validation_for_full_eval),
        "model_source": model_source,
        "resolved_base_checkpoint": resolved_base_checkpoint,
        "final_model_dir": str(final_model_dir),
        "best_full_eval_dir": callback.state.best_model_dir,
        "best_full_eval_metrics_path": callback.state.best_metrics_path,
    }
    save_json(output_dir / "effective_config.json", effective_config)

    if callback.state.best_model_dir and callback.state.best_metrics_path:
        with open(callback.state.best_metrics_path, "r", encoding="utf-8") as handle:
            best_metrics_payload = json.load(handle)
        summary = {
            "stage": args.stage,
            "preset": args.preset,
            "output_dir": str(output_dir),
            "best_model_dir": callback.state.best_model_dir,
            "best_metrics": best_metrics_payload["metrics"],
            "best_epoch": callback.state.best_epoch,
            "trainer_metrics": train_metrics,
        }
        save_json(output_dir / "run_summary.json", summary)


if __name__ == "__main__":
    main()
