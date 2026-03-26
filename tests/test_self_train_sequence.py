from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_self_train_sequence import _build_sequence_summary, _iter_manifests, _resolve_baseline_summary
from whisper_pularr.sequence_policy import beats_reference


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class SelfTrainSequenceTests(unittest.TestCase):
    def test_iter_manifests_can_select_final_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifests_dir = Path(tmp)
            for name in ("pseudo_labels_001000.jsonl", "pseudo_labels_002000.jsonl", "pseudo_labels_final.jsonl"):
                (manifests_dir / name).write_text("", encoding="utf-8")

            manifests = _iter_manifests(manifests_dir, final_only=True)
            self.assertEqual([path.name for path in manifests], ["pseudo_labels_final.jsonl"])

    def test_iter_manifests_can_limit_to_latest_n(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifests_dir = Path(tmp)
            for name in ("pseudo_labels_001000.jsonl", "pseudo_labels_002000.jsonl", "pseudo_labels_003000.jsonl"):
                (manifests_dir / name).write_text("", encoding="utf-8")

            manifests = _iter_manifests(manifests_dir, max_manifests=2)
            self.assertEqual([path.name for path in manifests], ["pseudo_labels_002000.jsonl", "pseudo_labels_003000.jsonl"])

    def test_beats_reference_uses_cer_as_tiebreak(self) -> None:
        reference = {"normalized_wer": 0.46, "normalized_cer": 0.13}
        better = {"normalized_wer": 0.45, "normalized_cer": 0.2}
        tied_better_cer = {"normalized_wer": 0.46, "normalized_cer": 0.12}
        tied_worse_cer = {"normalized_wer": 0.46, "normalized_cer": 0.14}

        self.assertTrue(beats_reference(better, reference))
        self.assertTrue(beats_reference(tied_better_cer, reference))
        self.assertFalse(beats_reference(tied_worse_cer, reference))

    def test_worse_than_baseline_is_rejected_but_sequence_can_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests_dir = root / "manifests"
            output_root = root / "runs"
            manifests_dir.mkdir()
            output_root.mkdir()

            manifests = []
            for name in ["pseudo_labels_001000.jsonl", "pseudo_labels_002000.jsonl"]:
                path = manifests_dir / name
                path.write_text("", encoding="utf-8")
                manifests.append(path)

            run_dir = output_root / "pseudo_labels_001000"
            run_dir.mkdir()
            _save_json(
                run_dir / "run_summary.json",
                {
                    "best_epoch": 1.0,
                    "best_model_dir": str(run_dir / "best_full_eval"),
                    "best_metrics": {
                        "normalized_wer": 0.47,
                        "normalized_cer": 0.13,
                        "raw_wer": 0.5,
                        "raw_cer": 0.14,
                    },
                },
            )
            _save_json(
                run_dir / "sequence_launch.json",
                {
                    "base_checkpoint_used": "/baseline/best_full_eval",
                    "comparison_target": "/baseline/best_full_eval",
                },
            )

            summary = _build_sequence_summary(
                manifests_dir,
                manifests,
                output_root,
                baseline_checkpoint="/baseline/best_full_eval",
                baseline_metrics={
                    "normalized_wer": 0.46,
                    "normalized_cer": 0.13,
                    "raw_wer": 0.49,
                    "raw_cer": 0.14,
                },
            )

            entry = summary["runs"][0]
            self.assertEqual(entry["status"], "rejected")
            self.assertFalse(entry["promoted"])
            self.assertEqual(summary["last_best_full_eval_dir"], "/baseline/best_full_eval")
            self.assertFalse(summary["stop_requested"])
            self.assertEqual(summary["runs"][1]["status"], "pending")

    def test_better_snapshot_is_promoted_and_becomes_next_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests_dir = root / "manifests"
            output_root = root / "runs"
            manifests_dir.mkdir()
            output_root.mkdir()

            manifests = []
            for name in ["pseudo_labels_001000.jsonl", "pseudo_labels_002000.jsonl"]:
                path = manifests_dir / name
                path.write_text("", encoding="utf-8")
                manifests.append(path)

            first_run = output_root / "pseudo_labels_001000"
            first_run.mkdir()
            _save_json(
                first_run / "run_summary.json",
                {
                    "best_epoch": 1.0,
                    "best_model_dir": str(first_run / "best_full_eval"),
                    "best_metrics": {
                        "normalized_wer": 0.45,
                        "normalized_cer": 0.12,
                        "raw_wer": 0.49,
                        "raw_cer": 0.14,
                    },
                },
            )
            _save_json(
                first_run / "sequence_launch.json",
                {
                    "base_checkpoint_used": "/baseline/best_full_eval",
                    "comparison_target": "/baseline/best_full_eval",
                },
            )

            summary = _build_sequence_summary(
                manifests_dir,
                manifests,
                output_root,
                baseline_checkpoint="/baseline/best_full_eval",
                baseline_metrics={
                    "normalized_wer": 0.46,
                    "normalized_cer": 0.13,
                    "raw_wer": 0.49,
                    "raw_cer": 0.14,
                },
            )

            first_entry, second_entry = summary["runs"]
            self.assertEqual(first_entry["status"], "completed")
            self.assertTrue(first_entry["promoted"])
            self.assertEqual(summary["last_best_full_eval_dir"], str(first_run / "best_full_eval"))
            self.assertEqual(second_entry["status"], "pending")
            self.assertEqual(second_entry["base_checkpoint_used"], str(first_run / "best_full_eval"))

    def test_resolve_baseline_summary_reads_best_eval_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "trial" / "best_full_eval"
            checkpoint_dir.mkdir(parents=True)
            _save_json(
                checkpoint_dir.parent / "best_full_eval_summary.json",
                {
                    "best_normalized_wer": 23.5,
                    "best_normalized_cer": 9.5,
                },
            )

            checkpoint, metrics = _resolve_baseline_summary(str(checkpoint_dir))
            self.assertEqual(checkpoint, str(checkpoint_dir))
            self.assertEqual(metrics["normalized_wer"], 0.235)
            self.assertEqual(metrics["normalized_cer"], 0.095)


if __name__ == "__main__":
    unittest.main()
