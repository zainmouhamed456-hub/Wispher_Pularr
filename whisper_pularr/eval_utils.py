from __future__ import annotations

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
    from transformers import TrainerCallback, pipeline
    from transformers.modeling_utils import unwrap_model
except ImportError:
    TrainerCallback = object
    pipeline = None

    def unwrap_model(model: Any) -> Any:
        return model
try:
    from .data import save_json
except ImportError:
    def save_json(path: str | Path, payload: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
from .settings import (
    DEFAULT_EARLY_STOP_PATIENCE_EPOCHS,
    DEFAULT_EVAL_CHUNK_LENGTH_SECONDS,
    DEFAULT_HARD_STOP_CER,
    DEFAULT_HARD_STOP_WER,
    DEFAULT_STOP_CER,
    DEFAULT_STOP_WER,
)
from .text import normalize_transcript


def _edit_distance(source: list[str], target: list[str]) -> int:
    previous = list(range(len(target) + 1))
    for source_index, source_item in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_item in enumerate(target, start=1):
            substitution_cost = 0 if source_item == target_item else 1
            current.append(
                min(
                    previous[target_index] + 1,
                    current[target_index - 1] + 1,
                    previous[target_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _sequence_error_rate(references: list[list[str]], predictions: list[list[str]]) -> float:
    total_distance = 0
    total_units = 0
    for reference, prediction in zip(references, predictions):
        total_distance += _edit_distance(reference, prediction)
        total_units += len(reference)
    if total_units == 0:
        return 0.0
    return float(total_distance / total_units)


def compute_error_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    normalized_references = [normalize_transcript(text) for text in references]
    normalized_predictions = [normalize_transcript(text) for text in predictions]
    if jiwer is None:
        raw_wer = _sequence_error_rate([text.split() for text in references], [text.split() for text in predictions])
        raw_cer = _sequence_error_rate([list(text) for text in references], [list(text) for text in predictions])
        normalized_wer = _sequence_error_rate(
            [text.split() for text in normalized_references],
            [text.split() for text in normalized_predictions],
        )
        normalized_cer = _sequence_error_rate(
            [list(text) for text in normalized_references],
            [list(text) for text in normalized_predictions],
        )
        return {
            "raw_wer": raw_wer,
            "raw_cer": raw_cer,
            "normalized_wer": normalized_wer,
            "normalized_cer": normalized_cer,
        }
    return {
        "raw_wer": float(jiwer.wer(references, predictions)),
        "raw_cer": float(jiwer.cer(references, predictions)),
        "normalized_wer": float(jiwer.wer(normalized_references, normalized_predictions)),
        "normalized_cer": float(jiwer.cer(normalized_references, normalized_predictions)),
    }


def _pipeline_dtype(runtime_config: Any) -> torch.dtype:
    if torch is None:
        return None
    if torch.cuda.is_available():
        if getattr(runtime_config, "bf16", False):
            return torch.bfloat16
        if getattr(runtime_config, "fp16", False):
            return torch.float16
    return torch.float32


def evaluate_long_form_dataset(
    model: Any,
    processor: Any,
    dataset: Any,
    runtime_config: Any,
    output_path: str | None = None,
    language: str | None = None,
    chunk_length_s: int = DEFAULT_EVAL_CHUNK_LENGTH_SECONDS,
) -> dict[str, Any]:
    if pipeline is None or torch is None:
        raise RuntimeError("transformers and torch are required for long-form evaluation.")
    bare_model = unwrap_model(model)
    bare_model.eval()
    device = 0 if torch.cuda.is_available() else -1
    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=bare_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_length_s,
        batch_size=getattr(runtime_config, "evaluation_batch_size", 8),
        torch_dtype=_pipeline_dtype(runtime_config),
        device=device,
    )
    generate_kwargs: dict[str, Any] = {}
    num_beams = int(getattr(runtime_config, "generation_num_beams", 1) or 1)
    if num_beams > 1:
        generate_kwargs["num_beams"] = num_beams
    if language:
        generate_kwargs["language"] = language

    references: list[str] = []
    predictions: list[str] = []
    per_sample: list[dict[str, Any]] = []
    batch_size = max(int(getattr(runtime_config, "evaluation_batch_size", 8)), 1)
    rows = list(dataset)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_inputs = [
            {"raw": row["audio"]["array"], "sampling_rate": row["audio"]["sampling_rate"]}
            for row in batch_rows
        ]
        batch_results = asr_pipeline(
            batch_inputs,
            generate_kwargs=generate_kwargs if generate_kwargs else None,
            return_timestamps=False,
        )
        if isinstance(batch_results, dict):
            batch_results = [batch_results]
        for row, result in zip(batch_rows, batch_results):
            prediction = (result.get("text") or "").strip()
            reference = (row.get("transcription") or "").strip()
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

    metrics = compute_error_metrics(references, predictions)
    payload = {
        "sample_count": len(per_sample),
        "metrics": metrics,
        "samples": per_sample,
    }
    if output_path:
        save_json(output_path, payload)
    return payload


@dataclass
class CallbackState:
    best_normalized_wer: float = float("inf")
    best_normalized_cer: float = float("inf")
    best_epoch: float | None = None
    best_metrics_path: str | None = None
    best_model_dir: str | None = None
    epochs_without_improvement: int = 0


class LongFormEvalAndStopCallback(TrainerCallback):
    def __init__(
        self,
        processor: Any,
        eval_dataset: Any,
        runtime_config: Any,
        output_dir: str,
        language: str | None = None,
        stop_wer: float = DEFAULT_STOP_WER,
        stop_cer: float = DEFAULT_STOP_CER,
        hard_stop_wer: float = DEFAULT_HARD_STOP_WER,
        hard_stop_cer: float = DEFAULT_HARD_STOP_CER,
        early_stop_patience_epochs: int = DEFAULT_EARLY_STOP_PATIENCE_EPOCHS,
    ) -> None:
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.runtime_config = runtime_config
        self.output_dir = Path(output_dir)
        self.language = language
        self.stop_wer = stop_wer
        self.stop_cer = stop_cer
        self.hard_stop_wer = hard_stop_wer
        self.hard_stop_cer = hard_stop_cer
        self.early_stop_patience_epochs = max(int(early_stop_patience_epochs), 0)
        self.state = CallbackState()

    def _save_best_model(self, model: Any) -> str:
        best_dir = self.output_dir / "best_full_eval"
        best_dir.mkdir(parents=True, exist_ok=True)
        unwrap_model(model).save_pretrained(best_dir)
        self.processor.save_pretrained(best_dir)
        return str(best_dir)

    def _write_summary(self) -> None:
        summary_path = self.output_dir / "best_full_eval_summary.json"
        save_json(
            summary_path,
            {
                "best_epoch": self.state.best_epoch,
                "best_normalized_wer": self.state.best_normalized_wer,
                "best_normalized_cer": self.state.best_normalized_cer,
                "best_metrics_path": self.state.best_metrics_path,
                "best_model_dir": self.state.best_model_dir,
                "epochs_without_improvement": self.state.epochs_without_improvement,
            },
        )

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control

        epoch = int(state.epoch or 0)
        metrics_path = self.output_dir / "full_validation" / f"epoch_{epoch:02d}.json"
        payload = evaluate_long_form_dataset(
            model=model,
            processor=self.processor,
            dataset=self.eval_dataset,
            runtime_config=self.runtime_config,
            output_path=str(metrics_path),
            language=self.language,
        )
        metrics = payload["metrics"]
        normalized_wer = float(metrics["normalized_wer"]) * 100.0
        normalized_cer = float(metrics["normalized_cer"]) * 100.0

        is_better = (
            normalized_wer < self.state.best_normalized_wer
            or (
                normalized_wer == self.state.best_normalized_wer
                and normalized_cer < self.state.best_normalized_cer
            )
        )
        if is_better:
            self.state.best_normalized_wer = normalized_wer
            self.state.best_normalized_cer = normalized_cer
            self.state.best_epoch = float(state.epoch or 0)
            self.state.best_metrics_path = str(metrics_path)
            self.state.best_model_dir = self._save_best_model(model)
            self.state.epochs_without_improvement = 0
            self._write_summary()
        else:
            self.state.epochs_without_improvement += 1

        if normalized_wer <= self.hard_stop_wer and normalized_cer <= self.hard_stop_cer:
            control.should_training_stop = True
        elif normalized_wer <= self.stop_wer and normalized_cer <= self.stop_cer:
            control.should_training_stop = True
        elif self.early_stop_patience_epochs and self.state.epochs_without_improvement >= self.early_stop_patience_epochs:
            control.should_training_stop = True

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._write_summary()
        return control
