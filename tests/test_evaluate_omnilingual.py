from __future__ import annotations

import unittest

from evaluate_omnilingual import evaluate_omnilingual_dataset


class _DummyOmniPipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, inputs, *, lang=None, batch_size=2):
        self.calls.append({"inputs": inputs, "lang": lang, "batch_size": batch_size})
        return ["mi yidi jam" for _ in inputs]


class EvaluateOmnilingualTests(unittest.TestCase):
    def test_evaluate_omnilingual_dataset_computes_metrics_and_passes_lang(self) -> None:
        pipeline = _DummyOmniPipeline()
        payload = evaluate_omnilingual_dataset(
            asr_pipeline=pipeline,
            dataset=[
                {
                    "id": "1",
                    "audio": {"array": [0.0, 0.1], "sampling_rate": 16000},
                    "transcription": "mi yidi jam",
                },
                {
                    "id": "2",
                    "audio": {"array": [0.0, 0.1], "sampling_rate": 16000},
                    "transcription": "mi yidi jam",
                },
            ],
            lang="ful_Latn",
            batch_size=1,
        )

        self.assertEqual(payload["sample_count"], 2)
        self.assertEqual(payload["metrics"]["normalized_wer"], 0.0)
        self.assertEqual(len(pipeline.calls), 2)
        self.assertEqual(pipeline.calls[0]["lang"], ["ful_Latn"])
        self.assertEqual(pipeline.calls[0]["inputs"][0]["sample_rate"], 16000)
        self.assertIn("waveform", pipeline.calls[0]["inputs"][0])


if __name__ == "__main__":
    unittest.main()
