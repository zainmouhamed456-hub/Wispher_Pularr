from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from whisper_pularr.colab_t4_policy import (
    resolve_colab_base_model,
    resolve_launcher_settings,
    should_promote_checkpoint,
)


class ColabT4PolicyTests(unittest.TestCase):
    def test_resolve_base_model_prefers_explicit_env_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            runs_root = Path(tmp) / "runs"
            root.mkdir(parents=True)
            runs_root.mkdir(parents=True)

            resolved = resolve_colab_base_model(
                root=root,
                runs_root=runs_root,
                explicit_base_model="custom/checkpoint",
                default_model_id="openai/whisper-small",
            )

        self.assertEqual(resolved, "custom/checkpoint")

    def test_resolve_base_model_uses_promoted_checkpoint_before_downloaded_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            runs_root = Path(tmp) / "runs"
            promoted_checkpoint = runs_root / "runs" / "colab_supervised" / "session_001" / "best_full_eval"
            downloaded_checkpoint = root / "downloads" / "trial_a_best_full_eval" / "best_full_eval"
            promotion_summary = runs_root / "reports" / "colab_promotion_summary.json"
            promoted_checkpoint.mkdir(parents=True)
            downloaded_checkpoint.mkdir(parents=True)
            promotion_summary.parent.mkdir(parents=True)
            promotion_summary.write_text(
                json.dumps({"best_checkpoint": str(promoted_checkpoint), "best_metrics": {"normalized_wer": 0.4, "normalized_cer": 0.1}}),
                encoding="utf-8",
            )

            resolved = resolve_colab_base_model(
                root=root,
                runs_root=runs_root,
                explicit_base_model=None,
                default_model_id="openai/whisper-small",
                promotion_summary_path=promotion_summary,
            )

        self.assertEqual(resolved, str(promoted_checkpoint))

    def test_resolve_launcher_settings_parses_eval_only_and_beam_envs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            runs_root = Path(tmp) / "runs"
            (root / "downloads" / "trial_a_best_full_eval" / "best_full_eval").mkdir(parents=True)
            settings = resolve_launcher_settings(
                {
                    "COLAB_EVAL_ONLY": "1",
                    "COLAB_COMPARE_BEAMS": "1, 3,3",
                    "COLAB_FIXED_SLICE_SIZE": "96",
                    "COLAB_RESUME_FROM": "runs/colab_supervised/session_001/checkpoint-125",
                },
                root=root,
                runs_root=runs_root,
                default_model_id="openai/whisper-small",
            )

        self.assertTrue(settings.eval_only)
        self.assertEqual(settings.compare_beams, (1, 3))
        self.assertEqual(settings.fixed_slice_size, 96)
        self.assertEqual(settings.resume_from, "runs/colab_supervised/session_001/checkpoint-125")
        self.assertEqual(settings.base_model, str(root / "downloads" / "trial_a_best_full_eval" / "best_full_eval"))

    def test_should_promote_requires_one_point_wer_improvement_or_cer_tiebreak(self) -> None:
        reference = {"normalized_wer": 0.4607, "normalized_cer": 0.1290}
        small_improvement = {"normalized_wer": 0.4550, "normalized_cer": 0.1000}
        tied_wer_better_cer = {"normalized_wer": 0.4607, "normalized_cer": 0.1200}
        full_point_improvement = {"normalized_wer": 0.4500, "normalized_cer": 0.2000}

        promote_small, _ = should_promote_checkpoint(small_improvement, reference)
        promote_tied, reason_tied = should_promote_checkpoint(tied_wer_better_cer, reference)
        promote_full, reason_full = should_promote_checkpoint(full_point_improvement, reference)

        self.assertFalse(promote_small)
        self.assertTrue(promote_tied)
        self.assertEqual(reason_tied, "wer_tied_cer_improved")
        self.assertTrue(promote_full)
        self.assertEqual(reason_full, "wer_improved")


if __name__ == "__main__":
    unittest.main()
