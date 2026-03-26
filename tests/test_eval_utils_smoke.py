from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
