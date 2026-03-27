from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class EntrypointHelpTests(unittest.TestCase):
    def _assert_help(self, script_name: str) -> None:
        completed = subprocess.run(
            [sys.executable, script_name, "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
        self.assertIn("usage:", completed.stdout.lower())

    def test_train_help(self) -> None:
        self._assert_help("train.py")

    def test_evaluate_help(self) -> None:
        self._assert_help("evaluate.py")

    def test_pseudo_label_help(self) -> None:
        self._assert_help("pseudo_label.py")

    def test_compare_checkpoints_help(self) -> None:
        self._assert_help("compare_checkpoints.py")

    def test_analyze_eval_help(self) -> None:
        self._assert_help("analyze_eval.py")

    def test_run_remote_pipeline_help(self) -> None:
        self._assert_help("run_remote_pipeline.py")

    def test_run_self_train_sequence_help(self) -> None:
        self._assert_help("run_self_train_sequence.py")

    def test_dashboard_help(self) -> None:
        self._assert_help("dashboard.py")

    def test_colab_run_t4_free_help(self) -> None:
        self._assert_help(str(Path("colab") / "run_t4_free.py"))


if __name__ == "__main__":
    unittest.main()
