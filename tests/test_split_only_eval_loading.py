from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import compare_checkpoints
import evaluate


class _FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def select(self, indices) -> "_FakeDataset":
        return _FakeDataset([self.rows[index] for index in list(indices)])

    def __len__(self) -> int:
        return len(self.rows)


class SplitOnlyEvalLoadingTests(unittest.TestCase):
    def test_evaluate_main_loads_requested_split_only(self) -> None:
        dataset = _FakeDataset([{"id": "1", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "ok"}])
        args = Namespace(
            checkpoint="checkpoint",
            dataset_name="google/WaxalNLP",
            dataset_config="ful_asr",
            split="validation",
            whisper_language=None,
            hardware_report=None,
            cache_dir=None,
            output_path=None,
            max_samples=None,
            generation_num_beams=None,
            evaluation_batch_size=None,
            chunk_length_s=None,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            fake_processor = SimpleNamespace(tokenizer=object())
            with (
                mock.patch.object(evaluate, "_require_eval_runtime", return_value=None),
                mock.patch.object(evaluate, "parse_args", return_value=args),
                mock.patch.object(evaluate, "runtime_from_optional_report", return_value=SimpleNamespace(cache_root=str(output_dir))),
                mock.patch.object(evaluate, "load_asr_split", return_value=dataset) as mocked_load_split,
                mock.patch.object(evaluate, "AutoProcessor", SimpleNamespace(from_pretrained=lambda _checkpoint: fake_processor)),
                mock.patch.object(evaluate, "AutoModelForSpeechSeq2Seq", SimpleNamespace(from_pretrained=lambda _checkpoint: object())),
                mock.patch.object(evaluate, "resolve_whisper_language", return_value=None),
                mock.patch.object(evaluate, "configure_whisper_prompt", return_value=None),
                mock.patch.object(
                    evaluate,
                    "evaluate_long_form_dataset",
                    return_value={"metrics": {"wer": 0.1, "cer": 0.05}, "sample_count": 1},
                ),
                mock.patch.object(evaluate, "save_json", return_value=None),
            ):
                evaluate.main()

        mocked_load_split.assert_called_once_with(
            "google/WaxalNLP",
            "ful_asr",
            split="validation",
            cache_dir=str(output_dir),
        )

    def test_compare_main_loads_requested_split_only(self) -> None:
        dataset = _FakeDataset(
            [
                {"id": "1", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "ok"},
                {"id": "2", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "ok"},
            ]
        )
        args = Namespace(
            checkpoints=["checkpoint-a"],
            dataset_name="google/WaxalNLP",
            dataset_config="ful_asr",
            split="validation",
            fixed_slice_size=1,
            whisper_language=None,
            hardware_report=None,
            cache_dir=None,
            output_dir="unused",
            skip_full=True,
            generation_num_beams=None,
            evaluation_batch_size=None,
            chunk_length_s=None,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            args.output_dir = str(output_dir)
            with (
                mock.patch.object(compare_checkpoints, "_require_compare_runtime", return_value=None),
                mock.patch.object(compare_checkpoints, "parse_args", return_value=args),
                mock.patch.object(
                    compare_checkpoints,
                    "runtime_from_optional_report",
                    return_value=SimpleNamespace(cache_root=str(output_dir)),
                ),
                mock.patch.object(compare_checkpoints, "load_asr_split", return_value=dataset) as mocked_load_split,
                mock.patch.object(
                    compare_checkpoints,
                    "_evaluate_checkpoint",
                    return_value={"checkpoint": "checkpoint-a", "fixed_slice_metrics": {}, "full_metrics": None},
                ),
                mock.patch.object(compare_checkpoints, "save_json", return_value=None),
            ):
                compare_checkpoints.main()

        mocked_load_split.assert_called_once_with(
            "google/WaxalNLP",
            "ful_asr",
            split="validation",
            cache_dir=str(output_dir),
        )

    def test_evaluate_main_passes_decode_overrides(self) -> None:
        dataset = _FakeDataset([{"id": "1", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "ok"}])
        args = Namespace(
            checkpoint="checkpoint",
            dataset_name="google/WaxalNLP",
            dataset_config="ful_asr",
            split="validation",
            whisper_language=None,
            hardware_report=None,
            cache_dir=None,
            output_path="custom.json",
            max_samples=None,
            generation_num_beams=3,
            evaluation_batch_size=2,
            chunk_length_s=18,
        )
        observed_kwargs: dict[str, object] = {}

        def _fake_eval(**kwargs):
            observed_kwargs.update(kwargs)
            return {"metrics": {"wer": 0.1, "cer": 0.05}, "sample_count": 1}

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            fake_processor = SimpleNamespace(tokenizer=object())
            with (
                mock.patch.object(evaluate, "_require_eval_runtime", return_value=None),
                mock.patch.object(evaluate, "parse_args", return_value=args),
                mock.patch.object(evaluate, "runtime_from_optional_report", return_value=SimpleNamespace(cache_root=str(output_dir))),
                mock.patch.object(evaluate, "load_asr_split", return_value=dataset),
                mock.patch.object(evaluate, "AutoProcessor", SimpleNamespace(from_pretrained=lambda _checkpoint: fake_processor)),
                mock.patch.object(evaluate, "AutoModelForSpeechSeq2Seq", SimpleNamespace(from_pretrained=lambda _checkpoint: object())),
                mock.patch.object(evaluate, "resolve_whisper_language", return_value=None),
                mock.patch.object(evaluate, "configure_whisper_prompt", return_value=None),
                mock.patch.object(evaluate, "evaluate_long_form_dataset", side_effect=_fake_eval),
                mock.patch.object(evaluate, "save_json", return_value=None),
            ):
                evaluate.main()

        self.assertEqual(observed_kwargs["generation_num_beams"], 3)
        self.assertEqual(observed_kwargs["evaluation_batch_size"], 2)
        self.assertEqual(observed_kwargs["chunk_length_s"], 18)

    def test_compare_main_passes_decode_overrides(self) -> None:
        dataset = _FakeDataset(
            [
                {"id": "1", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "ok"},
                {"id": "2", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "ok"},
            ]
        )
        args = Namespace(
            checkpoints=["checkpoint-a"],
            dataset_name="google/WaxalNLP",
            dataset_config="ful_asr",
            split="validation",
            fixed_slice_size=1,
            whisper_language=None,
            hardware_report=None,
            cache_dir=None,
            output_dir="unused",
            skip_full=True,
            generation_num_beams=5,
            evaluation_batch_size=4,
            chunk_length_s=22,
        )
        observed_kwargs: dict[str, object] = {}

        def _fake_evaluate_checkpoint(checkpoint, **kwargs):
            observed_kwargs["checkpoint"] = checkpoint
            observed_kwargs.update(kwargs)
            return {"checkpoint": "checkpoint-a", "fixed_slice_metrics": {}, "full_metrics": None}

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            args.output_dir = str(output_dir)
            with (
                mock.patch.object(compare_checkpoints, "_require_compare_runtime", return_value=None),
                mock.patch.object(compare_checkpoints, "parse_args", return_value=args),
                mock.patch.object(
                    compare_checkpoints,
                    "runtime_from_optional_report",
                    return_value=SimpleNamespace(cache_root=str(output_dir)),
                ),
                mock.patch.object(compare_checkpoints, "load_asr_split", return_value=dataset),
                mock.patch.object(compare_checkpoints, "_evaluate_checkpoint", side_effect=_fake_evaluate_checkpoint),
                mock.patch.object(compare_checkpoints, "save_json", return_value=None),
            ):
                compare_checkpoints.main()

        self.assertEqual(observed_kwargs["generation_num_beams"], 5)
        self.assertEqual(observed_kwargs["evaluation_batch_size"], 4)
        self.assertEqual(observed_kwargs["chunk_length_s"], 22)


if __name__ == "__main__":
    unittest.main()
