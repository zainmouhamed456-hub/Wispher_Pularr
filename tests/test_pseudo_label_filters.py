from __future__ import annotations

import unittest

from whisper_pularr.pseudo_label_policy import build_label_profile, evaluate_pseudo_label_record


class PseudoLabelFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        split = {
            "transcription": [
                "Mi yidi Pulaar e jam.",
                "Ko fii jango e leyde.",
                "Enen woni e suudu.",
            ]
        }
        self.label_profile = build_label_profile(split)
        self.base_kwargs = {
            "token_vocab": self.label_profile["token_vocab"],
            "allowed_chars": self.label_profile["allowed_chars"],
            "min_chars": 3,
            "avg_logprob_threshold": -0.6,
            "compression_ratio_threshold": 1.8,
            "no_speech_prob_threshold": 0.2,
            "min_labeled_token_ratio": 0.65,
            "min_labeled_char_ratio": 0.95,
        }

    def test_valid_pulaar_like_text_passes(self) -> None:
        record = {
            "pseudo_transcription": "Mi yidi jam e leyde.",
            "avg_logprob": -0.3,
            "compression_ratio": 1.1,
            "no_speech_prob": 0.01,
        }
        accepted, reasons = evaluate_pseudo_label_record(record, **self.base_kwargs)
        self.assertTrue(accepted)
        self.assertEqual(reasons, set())

    def test_english_like_text_fails_token_overlap(self) -> None:
        record = {
            "pseudo_transcription": "Thank you for the update.",
            "avg_logprob": -0.2,
            "compression_ratio": 1.0,
            "no_speech_prob": 0.01,
        }
        accepted, reasons = evaluate_pseudo_label_record(record, **self.base_kwargs)
        self.assertFalse(accepted)
        self.assertIn("language_mismatch", reasons)

    def test_non_latin_text_fails_char_overlap(self) -> None:
        record = {
            "pseudo_transcription": "දැන් අපි යමු",
            "avg_logprob": -0.2,
            "compression_ratio": 1.0,
            "no_speech_prob": 0.01,
        }
        accepted, reasons = evaluate_pseudo_label_record(record, **self.base_kwargs)
        self.assertFalse(accepted)
        self.assertIn("language_mismatch", reasons)

    def test_quality_thresholds_report_reason_codes(self) -> None:
        record = {
            "pseudo_transcription": "Mi",
            "avg_logprob": -0.9,
            "compression_ratio": 2.0,
            "no_speech_prob": 0.5,
        }
        accepted, reasons = evaluate_pseudo_label_record(record, **self.base_kwargs)
        self.assertFalse(accepted)
        self.assertIn("too_short", reasons)
        self.assertIn("low_logprob", reasons)
        self.assertIn("high_compression", reasons)
        self.assertIn("high_no_speech", reasons)


if __name__ == "__main__":
    unittest.main()
