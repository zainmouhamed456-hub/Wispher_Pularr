from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import train
from whisper_pularr.runtime import RuntimeConfig


class _FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self._column_names = list(rows[0].keys()) if rows else ["id", "audio", "transcription", "duration_seconds"]

    @property
    def column_names(self) -> list[str]:
        return self._column_names

    def select(self, indices) -> "_FakeDataset":
        return _FakeDataset([self.rows[index] for index in list(indices)])

    def shuffle(self, seed: int | None = None) -> "_FakeDataset":
        del seed
        return self

    def filter(self, predicate, desc: str | None = None) -> "_FakeDataset":
        del desc
        return _FakeDataset([row for row in self.rows if predicate(row)])

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row.get(key) for row in self.rows]
        return self.rows[key]

    def __len__(self) -> int:
        return len(self.rows)


class _FakeTrainer:
    train_kwargs_history: list[dict[str, object]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def train(self, **kwargs):
        self.__class__.train_kwargs_history.append(dict(kwargs))
        return SimpleNamespace(metrics={"train_runtime": 1.0})

    def save_model(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)


class _FakeCallback:
    def __init__(self, *, output_dir: str, **kwargs) -> None:
        del kwargs
        best_dir = Path(output_dir) / "best_full_eval"
        best_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = Path(output_dir) / "best_metrics.json"
        metrics_path.write_text(
            json.dumps({"metrics": {"normalized_wer": 0.4, "normalized_cer": 0.1, "raw_wer": 0.45, "raw_cer": 0.12}}),
            encoding="utf-8",
        )
        self.state = SimpleNamespace(
            best_model_dir=str(best_dir),
            best_metrics_path=str(metrics_path),
            best_epoch=1.0,
        )


def _fake_runtime() -> RuntimeConfig:
    return RuntimeConfig(
        bf16=False,
        fp16=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=0,
        pseudo_label_batch_size=1,
        evaluation_batch_size=1,
        generation_num_beams=1,
        save_steps=10,
        logging_steps=1,
        use_multi_gpu=False,
        cache_root="hf-cache",
        output_root="runs",
        profile="colab_t4",
    )


def _fake_args(output_dir: str, resume_from_checkpoint: str | None) -> Namespace:
    return Namespace(
        stage="supervised",
        preset="trial_a",
        dataset_name="google/WaxalNLP",
        dataset_config="ful_asr",
        aux_dataset_name=None,
        aux_dataset_config=None,
        aux_labeled_repeat_count=1,
        model_id="openai/whisper-small",
        whisper_language=None,
        base_checkpoint=None,
        resume_from_checkpoint=resume_from_checkpoint,
        pseudo_labels_path=None,
        hardware_report=None,
        cache_dir=None,
        output_dir=output_dir,
        num_train_epochs=1.0,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_label_length=256,
        max_train_duration_seconds=30.0,
        save_total_limit=1,
        seed=42,
        labeled_repeat_count=None,
        learning_rate=None,
        max_train_samples=None,
        max_eval_samples=None,
        early_stop_patience_epochs=1,
        streaming=False,
    )


def _fake_model_and_processor() -> tuple[SimpleNamespace, SimpleNamespace]:
    model = SimpleNamespace(
        generation_config=SimpleNamespace(num_beams=None),
        config=SimpleNamespace(use_cache=True, apply_spec_augment=False),
    )
    processor = SimpleNamespace(tokenizer=object(), save_pretrained=lambda output_dir: Path(output_dir).mkdir(parents=True, exist_ok=True))
    return model, processor


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))


class TrainResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeTrainer.train_kwargs_history.clear()

    def _run_main(self, *, resume_from_checkpoint: str | None) -> dict[str, object]:
        train_split = _FakeDataset(
            [{"id": "t1", "audio": {"array": [0], "sampling_rate": 1}, "transcription": "ok", "duration_seconds": 1.0}]
        )
        validation_split = _FakeDataset(
            [{"id": "v1", "audio": {"array": [0], "sampling_rate": 1}, "transcription": "ok", "duration_seconds": 1.0}]
        )
        with tempfile.TemporaryDirectory() as tmp:
            args = _fake_args(str(Path(tmp) / "run"), resume_from_checkpoint=resume_from_checkpoint)
            model, processor = _fake_model_and_processor()
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(train, "_require_train_runtime", return_value=None))
                stack.enter_context(mock.patch.object(train, "parse_args", return_value=args))
                stack.enter_context(mock.patch.object(train, "runtime_from_optional_report", return_value=_fake_runtime()))
                stack.enter_context(mock.patch.object(train, "runtime_for_stage", side_effect=lambda stage, runtime, use_cuda: runtime))
                stack.enter_context(mock.patch.object(train, "torch", _fake_torch()))
                stack.enter_context(mock.patch.object(train, "load_asr_split", side_effect=[train_split, validation_split]))
                stack.enter_context(mock.patch.object(train, "prepare_model_and_processor", return_value=(model, processor)))
                stack.enter_context(mock.patch.object(train, "resolve_base_checkpoint", return_value=None))
                stack.enter_context(mock.patch.object(train, "resolve_whisper_language", return_value=None))
                stack.enter_context(mock.patch.object(train, "configure_whisper_prompt", return_value=None))
                stack.enter_context(mock.patch.object(train, "build_stage_dataset", return_value=train_split))
                stack.enter_context(mock.patch.object(train, "normalize_supervised_split_schema", side_effect=lambda dataset: dataset))
                stack.enter_context(mock.patch.object(train, "filter_empty_transcriptions", side_effect=lambda dataset: dataset))
                stack.enter_context(mock.patch.object(train, "filter_by_max_duration", side_effect=lambda dataset, *_args, **_kwargs: dataset))
                stack.enter_context(mock.patch.object(train, "load_auxiliary_train_split", return_value=None))
                stack.enter_context(mock.patch.object(train, "preprocess_split", side_effect=lambda split, *args, **kwargs: split))
                stack.enter_context(mock.patch.object(train, "WhisperDataCollatorSpeechSeq2SeqWithPadding", return_value=object()))
                stack.enter_context(mock.patch.object(train, "Seq2SeqTrainingArguments", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)))
                stack.enter_context(mock.patch.object(train, "LongFormEvalAndStopCallback", side_effect=lambda **kwargs: _FakeCallback(**kwargs)))
                stack.enter_context(mock.patch.object(train, "Seq2SeqTrainer", side_effect=lambda **kwargs: _FakeTrainer(**kwargs)))
                stack.enter_context(mock.patch.object(train, "build_compute_metrics", return_value=None))
                train.main()

        return _FakeTrainer.train_kwargs_history[-1]

    def test_train_main_does_not_pass_resume_kwarg_by_default(self) -> None:
        observed_kwargs = self._run_main(resume_from_checkpoint=None)
        self.assertEqual(observed_kwargs, {})

    def test_train_main_passes_resume_kwarg_when_requested(self) -> None:
        observed_kwargs = self._run_main(resume_from_checkpoint="runs/session_001/checkpoint-125")
        self.assertEqual(observed_kwargs, {"resume_from_checkpoint": "runs/session_001/checkpoint-125"})


if __name__ == "__main__":
    unittest.main()
