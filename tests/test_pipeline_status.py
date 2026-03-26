from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from whisper_pularr.pipeline_status import select_best_supervised_checkpoint, sequence_summary_complete


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class PipelineStatusTests(unittest.TestCase):
    def test_select_best_supervised_checkpoint_uses_wer_then_cer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"
            for trial_name, wer, cer in (
                ("trial_a", 0.31, 0.11),
                ("trial_b", 0.28, 0.14),
                ("trial_c", 0.28, 0.10),
            ):
                checkpoint_dir = runs_root / trial_name / "best_full_eval"
                checkpoint_dir.mkdir(parents=True)
                _save_json(
                    runs_root / trial_name / "run_summary.json",
                    {
                        "best_metrics": {
                            "normalized_wer": wer,
                            "normalized_cer": cer,
                        }
                    },
                )

            best = select_best_supervised_checkpoint(runs_root)
            self.assertEqual(best, str(runs_root / "trial_c" / "best_full_eval"))

    def test_sequence_summary_complete_requires_final_manifest_and_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "sequence_summary.json"
            manifests_dir = root / "manifests"
            manifests_dir.mkdir()
            (manifests_dir / "pseudo_labels_001000.jsonl").write_text("", encoding="utf-8")
            (manifests_dir / "pseudo_labels_final.jsonl").write_text("", encoding="utf-8")

            _save_json(
                summary_path,
                {
                    "runs": [
                        {
                            "manifest_path": str(manifests_dir / "pseudo_labels_final.jsonl"),
                            "status": "completed",
                        }
                    ]
                },
            )

            self.assertTrue(sequence_summary_complete(summary_path, manifests_dir))

            _save_json(
                summary_path,
                {
                    "runs": [
                        {
                            "manifest_path": str(manifests_dir / "pseudo_labels_final.jsonl"),
                            "status": "running",
                        }
                    ]
                },
            )
            self.assertFalse(sequence_summary_complete(summary_path, manifests_dir))


if __name__ == "__main__":
    unittest.main()
