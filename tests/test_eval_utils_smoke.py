from __future__ import annotations

import contextlib
import io
import unittest

import whisper_pularr.eval_utils as eval_utils


class _DummyTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False


class _DummyRuntime:
    bf16 = False
    fp16 = False
    evaluation_batch_size = 2
    generation_num_beams = 3


class _DummyModel:
    def eval(self) -> None:
        return None


class _DummyProcessor:
    tokenizer = object()
    feature_extractor = object()


class _DummyHFLogging:
    def __init__(self) -> None:
        self.verbosity = 30
        self.calls: list[tuple[str, int | None]] = []

    def get_verbosity(self) -> int:
        self.calls.append(("get_verbosity", None))
        return self.verbosity

    def set_verbosity_error(self) -> None:
        self.calls.append(("set_verbosity_error", None))

    def set_verbosity(self, value: int) -> None:
        self.calls.append(("set_verbosity", value))


class EvalUtilsSmokeTests(unittest.TestCase):
    def test_evaluate_long_form_dataset_uses_fixed_generate_kwargs(self) -> None:
        original_pipeline = eval_utils.pipeline
        original_torch = eval_utils.torch
        original_unwrap_model = eval_utils.unwrap_model
        observed_calls: list[dict] = []

        def fake_pipeline(**kwargs):
            def _runner(batch_inputs, generate_kwargs=None, return_timestamps=False):
                observed_calls.append(
                    {
                        "batch_size": len(batch_inputs),
                        "generate_kwargs": generate_kwargs,
                        "return_timestamps": return_timestamps,
                    }
                )
                return [{"text": "mi yidi jam"}, {"text": "ko fii jango"}][: len(batch_inputs)]

            return _runner

        eval_utils.pipeline = fake_pipeline
        eval_utils.torch = _DummyTorch()
        eval_utils.unwrap_model = lambda model: model
        try:
            payload = eval_utils.evaluate_long_form_dataset(
                model=_DummyModel(),
                processor=_DummyProcessor(),
                dataset=[
                    {"id": "1", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "mi yidi jam"},
                    {"id": "2", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "ko fii jango"},
                ],
                runtime_config=_DummyRuntime(),
                output_path=None,
                language="fr",
            )
        finally:
            eval_utils.pipeline = original_pipeline
            eval_utils.torch = original_torch
            eval_utils.unwrap_model = original_unwrap_model

        self.assertEqual(payload["sample_count"], 2)
        self.assertEqual(payload["metrics"]["normalized_wer"], 0.0)
        self.assertEqual(observed_calls[0]["generate_kwargs"], {"num_beams": 3, "language": "fr"})
        self.assertFalse(observed_calls[0]["return_timestamps"])

    def test_evaluate_long_form_dataset_allows_explicit_decode_overrides(self) -> None:
        original_pipeline = eval_utils.pipeline
        original_torch = eval_utils.torch
        original_unwrap_model = eval_utils.unwrap_model
        observed_calls: list[dict] = []
        observed_pipeline_kwargs: list[dict] = []

        def fake_pipeline(**kwargs):
            observed_pipeline_kwargs.append(kwargs)

            def _runner(batch_inputs, generate_kwargs=None, return_timestamps=False):
                observed_calls.append(
                    {
                        "batch_size": len(batch_inputs),
                        "generate_kwargs": generate_kwargs,
                        "return_timestamps": return_timestamps,
                    }
                )
                return [{"text": "mi yidi jam"} for _ in batch_inputs]

            return _runner

        eval_utils.pipeline = fake_pipeline
        eval_utils.torch = _DummyTorch()
        eval_utils.unwrap_model = lambda model: model
        try:
            payload = eval_utils.evaluate_long_form_dataset(
                model=_DummyModel(),
                processor=_DummyProcessor(),
                dataset=[
                    {"id": "1", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "mi yidi jam"},
                    {"id": "2", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "mi yidi jam"},
                ],
                runtime_config=_DummyRuntime(),
                output_path=None,
                language=None,
                chunk_length_s=18,
                evaluation_batch_size=1,
                generation_num_beams=4,
            )
        finally:
            eval_utils.pipeline = original_pipeline
            eval_utils.torch = original_torch
            eval_utils.unwrap_model = original_unwrap_model

        self.assertEqual(payload["sample_count"], 2)
        self.assertEqual(observed_pipeline_kwargs[0]["chunk_length_s"], 18)
        self.assertEqual(observed_pipeline_kwargs[0]["batch_size"], 1)
        self.assertEqual(observed_calls[0]["generate_kwargs"], {"num_beams": 4})

    def test_evaluate_long_form_dataset_logs_progress_and_restores_transformers_verbosity(self) -> None:
        original_pipeline = eval_utils.pipeline
        original_torch = eval_utils.torch
        original_unwrap_model = eval_utils.unwrap_model
        original_hf_logging = eval_utils.hf_logging
        fake_hf_logging = _DummyHFLogging()

        def fake_pipeline(**kwargs):
            def _runner(batch_inputs, generate_kwargs=None, return_timestamps=False):
                return [{"text": "mi yidi jam"} for _ in batch_inputs]

            return _runner

        eval_utils.pipeline = fake_pipeline
        eval_utils.torch = _DummyTorch()
        eval_utils.unwrap_model = lambda model: model
        eval_utils.hf_logging = fake_hf_logging
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                payload = eval_utils.evaluate_long_form_dataset(
                    model=_DummyModel(),
                    processor=_DummyProcessor(),
                    dataset=[
                        {"id": "1", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "mi yidi jam"},
                        {"id": "2", "audio": {"array": [0.0], "sampling_rate": 16000}, "transcription": "mi yidi jam"},
                    ],
                    runtime_config=_DummyRuntime(),
                    output_path=None,
                    language=None,
                    evaluation_batch_size=1,
                )
        finally:
            eval_utils.pipeline = original_pipeline
            eval_utils.torch = original_torch
            eval_utils.unwrap_model = original_unwrap_model
            eval_utils.hf_logging = original_hf_logging

        self.assertEqual(payload["sample_count"], 2)
        output = stdout.getvalue()
        self.assertIn("Evaluating 2 sample(s)", output)
        self.assertIn("Long-form eval progress: 1/2 sample(s)", output)
        self.assertIn("Long-form eval progress: 2/2 sample(s)", output)
        self.assertEqual(
            fake_hf_logging.calls,
            [("get_verbosity", None), ("set_verbosity_error", None), ("set_verbosity", 30)],
        )


if __name__ == "__main__":
    unittest.main()
