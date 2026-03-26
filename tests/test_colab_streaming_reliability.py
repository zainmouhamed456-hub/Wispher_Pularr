from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import train
from whisper_pularr import data
from whisper_pularr.runtime import RuntimeConfig


class _FakeStreamingDataset:
    def __init__(self, rows: list[dict[str, object]], feature_names: list[str] | None = None) -> None:
        self._rows = rows
        names = feature_names or list(rows[0].keys())
        self.features = {name: object() for name in names}

    def __iter__(self):
        return iter(self._rows)


class _FakeMaterializedDataset:
    def __init__(self, rows: list[dict[str, object]], column_names: list[str] | None = None) -> None:
        self.rows = rows
        self._column_names = column_names or (list(rows[0].keys()) if rows else [])

    @property
    def column_names(self) -> list[str]:
        return self._column_names

    def cast_column(self, _name: str, _value: object) -> "_FakeMaterializedDataset":
        return self

    def select(self, indices) -> "_FakeMaterializedDataset":
        return _FakeMaterializedDataset([self.rows[index] for index in list(indices)], column_names=self._column_names)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row.get(key) for row in self.rows]
        return self.rows[key]

    def __len__(self) -> int:
        return len(self.rows)


class _FakeDatasetFactory:
    @staticmethod
    def from_list(rows: list[dict[str, object]]) -> _FakeMaterializedDataset:
        return _FakeMaterializedDataset(rows)

    @staticmethod
    def from_dict(columns: dict[str, list[object]]) -> _FakeMaterializedDataset:
        names = list(columns.keys())
        if not names:
            return _FakeMaterializedDataset([], column_names=[])
        row_count = len(columns[names[0]])
        rows = [{name: columns[name][index] for name in names} for index in range(row_count)]
        return _FakeMaterializedDataset(rows, column_names=names)


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


def _fake_args(output_dir: str) -> Namespace:
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
        max_train_samples=128,
        max_eval_samples=16,
        early_stop_patience_epochs=1,
        streaming=True,
    )


def _fake_model_and_processor() -> tuple[SimpleNamespace, SimpleNamespace]:
    model = SimpleNamespace(
        generation_config=SimpleNamespace(num_beams=None),
        config=SimpleNamespace(use_cache=True, apply_spec_augment=False),
    )
    processor = SimpleNamespace(tokenizer=object())
    return model, processor


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))


class ColabStreamingReliabilityTests(unittest.TestCase):
    def test_streaming_loader_skips_invalid_leading_rows_and_fills_limit(self) -> None:
        rows = [
            {"id": "too-long-1", "audio": {"array": [0] * 31, "sampling_rate": 1}, "transcription": "a"},
            {"id": "too-long-2", "audio": {"array": [0] * 45, "sampling_rate": 1}, "transcription": "b"},
            {"id": "valid-1", "audio": {"array": [0] * 5, "sampling_rate": 1}, "transcription": "c"},
            {"id": "valid-2", "audio": {"array": [0] * 7, "sampling_rate": 1}, "transcription": "d"},
            {"id": "valid-3", "audio": {"array": [0] * 9, "sampling_rate": 1}, "transcription": "e"},
        ]
        stream = _FakeStreamingDataset(rows)

        with (
            mock.patch.object(data, "_require_datasets", return_value=None),
            mock.patch.object(data, "load_dataset", return_value=stream),
            mock.patch.object(data, "Dataset", _FakeDatasetFactory),
            mock.patch.object(data, "Audio", lambda sampling_rate: {"sampling_rate": sampling_rate}),
        ):
            split = data.load_asr_split(
                "google/WaxalNLP",
                "ful_asr",
                split="train",
                streaming=True,
                materialize_limit=2,
                max_duration_seconds=30.0,
            )

        self.assertEqual(len(split), 2)
        self.assertEqual(split["id"], ["valid-1", "valid-2"])
        self.assertEqual(split["duration_seconds"], [5.0, 7.0])

    def test_train_main_exits_clearly_when_supervised_split_is_empty(self) -> None:
        empty_split = _FakeMaterializedDataset([], column_names=["id", "audio", "transcription", "duration_seconds"])
        validation_split = _FakeMaterializedDataset(
            [{"id": "v1", "audio": {"array": [0], "sampling_rate": 1}, "transcription": "ok", "duration_seconds": 1.0}],
            column_names=["id", "audio", "transcription", "duration_seconds"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            args = _fake_args(str(Path(tmp) / "run"))
            model, processor = _fake_model_and_processor()
            with (
                mock.patch.object(train, "_require_train_runtime", return_value=None),
                mock.patch.object(train, "parse_args", return_value=args),
                mock.patch.object(train, "runtime_from_optional_report", return_value=_fake_runtime()),
                mock.patch.object(train, "runtime_for_stage", side_effect=lambda stage, runtime, use_cuda: runtime),
                mock.patch.object(train, "torch", _fake_torch()),
                mock.patch.object(train, "load_asr_split", side_effect=[validation_split, validation_split]),
                mock.patch.object(train, "prepare_model_and_processor", return_value=(model, processor)),
                mock.patch.object(train, "resolve_base_checkpoint", return_value=None),
                mock.patch.object(train, "resolve_whisper_language", return_value=None),
                mock.patch.object(train, "configure_whisper_prompt", return_value=None),
                mock.patch.object(train, "build_stage_dataset", return_value=empty_split),
                mock.patch.object(train, "normalize_supervised_split_schema", side_effect=lambda dataset: dataset),
            ):
                with self.assertRaises(SystemExit) as raised:
                    train.main()

        self.assertIn("No supervised training samples remained after filtering", str(raised.exception))

    def test_train_main_exits_clearly_when_validation_split_is_empty(self) -> None:
        train_split = _FakeMaterializedDataset(
            [{"id": "t1", "audio": {"array": [0], "sampling_rate": 1}, "transcription": "ok", "duration_seconds": 1.0}],
            column_names=["id", "audio", "transcription", "duration_seconds"],
        )
        empty_split = _FakeMaterializedDataset([], column_names=["id", "audio", "transcription", "duration_seconds"])

        with tempfile.TemporaryDirectory() as tmp:
            args = _fake_args(str(Path(tmp) / "run"))
            model, processor = _fake_model_and_processor()
            with (
                mock.patch.object(train, "_require_train_runtime", return_value=None),
                mock.patch.object(train, "parse_args", return_value=args),
                mock.patch.object(train, "runtime_from_optional_report", return_value=_fake_runtime()),
                mock.patch.object(train, "runtime_for_stage", side_effect=lambda stage, runtime, use_cuda: runtime),
                mock.patch.object(train, "torch", _fake_torch()),
                mock.patch.object(train, "load_asr_split", side_effect=[train_split, empty_split]),
                mock.patch.object(train, "prepare_model_and_processor", return_value=(model, processor)),
                mock.patch.object(train, "resolve_base_checkpoint", return_value=None),
                mock.patch.object(train, "resolve_whisper_language", return_value=None),
                mock.patch.object(train, "configure_whisper_prompt", return_value=None),
                mock.patch.object(train, "build_stage_dataset", return_value=train_split),
                mock.patch.object(train, "normalize_supervised_split_schema", side_effect=lambda dataset: dataset),
                mock.patch.object(train, "load_auxiliary_train_split", return_value=None),
                mock.patch.object(train, "preprocess_split", side_effect=lambda split, *args, **kwargs: split),
                mock.patch.object(train, "filter_by_max_duration", side_effect=lambda split, *_args, **_kwargs: split),
            ):
                with self.assertRaises(SystemExit) as raised:
                    train.main()

        self.assertIn("No validation samples remained after filtering", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
