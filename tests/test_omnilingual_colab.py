from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from whisper_pularr.omnilingual_colab import (
    DEFAULT_OMNI_LANG,
    waxal_row_to_omnilingual_record,
    should_promote_omnilingual,
    write_language_distribution,
    write_omnilingual_dataset_assets,
    write_omnilingual_parquet_split,
)


def _row(transcription: str, samples: int = 1600) -> dict[str, object]:
    return {
        "id": "sample-1",
        "transcription": transcription,
        "audio": {
            "array": np.zeros(samples, dtype=np.float32),
            "sampling_rate": 16000,
        },
    }


class OmnilingualColabTests(unittest.TestCase):
    def test_waxal_row_to_omnilingual_record_filters_and_encodes_audio(self) -> None:
        record = waxal_row_to_omnilingual_record(_row("  Awa, Pularr!  "), split="dev")

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["text"], "awa pularr")
        self.assertEqual(record["audio_size"], 1600)
        self.assertEqual(record["corpus"], "waxal")
        self.assertEqual(record["split"], "dev")
        self.assertEqual(record["language"], DEFAULT_OMNI_LANG)
        self.assertIsInstance(record["audio_bytes"], list)
        self.assertGreater(len(record["audio_bytes"]), 0)
        self.assertLessEqual(min(record["audio_bytes"]), 127)
        self.assertGreaterEqual(max(record["audio_bytes"]), -128)

        self.assertIsNone(waxal_row_to_omnilingual_record(_row("   "), split="train"))
        self.assertIsNone(
            waxal_row_to_omnilingual_record(
                _row("too long", samples=16000 * 41),
                split="train",
                max_duration_seconds=40.0,
            )
        )

    def test_write_parquet_uses_expected_schema_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            version_root = Path(tmp) / "waxal_ful_pularr" / "version=0"
            records = [
                waxal_row_to_omnilingual_record(_row("Mi yidi Pularr"), split="train"),
                waxal_row_to_omnilingual_record(_row("Ko fii jam"), split="train"),
            ]
            clean_records = [record for record in records if record is not None]

            output_path, count, audio_size = write_omnilingual_parquet_split(
                clean_records,
                version_root=version_root,
                split="train",
            )

            self.assertEqual(count, 2)
            self.assertEqual(audio_size, 3200)
            self.assertEqual(
                output_path.parent,
                version_root / "corpus=waxal" / "split=train" / f"language={DEFAULT_OMNI_LANG}",
            )
            table = pq.read_table(output_path)
            self.assertTrue(
                set(["text", "audio_bytes", "audio_size", "corpus", "split", "language"]).issubset(
                    set(table.column_names)
                )
            )
            self.assertEqual(table["split"].to_pylist(), ["train", "train"])
            self.assertEqual(table["language"].to_pylist(), [DEFAULT_OMNI_LANG, DEFAULT_OMNI_LANG])

    def test_write_assets_creates_card_configs_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "artifacts" / "waxal_ful_pularr"
            summary_path = write_language_distribution(
                dataset_root=dataset_root,
                corpus="waxal",
                lang=DEFAULT_OMNI_LANG,
                total_audio_size=3600 * 16000,
            )
            card_path, train_config_path, eval_config_path = write_omnilingual_dataset_assets(
                runs_root=tmp,
                base_model="omniASR_CTC_300M_v2",
                num_steps=100,
            )

            self.assertIn("hours", summary_path.read_text(encoding="utf-8"))
            self.assertIn("waxal_ful_pularr", card_path.read_text(encoding="utf-8"))
            train_config = train_config_path.read_text(encoding="utf-8")
            self.assertIn('name: "omniASR_CTC_300M_v2"', train_config)
            self.assertIn('name: "omniASR_tokenizer_written_v2"', train_config)
            self.assertIn("num_steps: 100", train_config)
            self.assertIn("valid_split", eval_config_path.read_text(encoding="utf-8"))

    def test_should_promote_omnilingual_uses_wer_then_cer(self) -> None:
        current = {"normalized_wer": 0.40, "normalized_cer": 0.10}

        self.assertTrue(should_promote_omnilingual({"normalized_wer": 0.39, "normalized_cer": 0.20}, current)[0])
        self.assertTrue(should_promote_omnilingual({"normalized_wer": 0.40, "normalized_cer": 0.09}, current)[0])
        self.assertFalse(should_promote_omnilingual({"normalized_wer": 0.41, "normalized_cer": 0.01}, current)[0])
        self.assertFalse(should_promote_omnilingual({"normalized_wer": 0.40, "normalized_cer": 0.11}, current)[0])
        self.assertTrue(should_promote_omnilingual({"normalized_wer": 0.50, "normalized_cer": 0.50}, None)[0])


if __name__ == "__main__":
    unittest.main()
