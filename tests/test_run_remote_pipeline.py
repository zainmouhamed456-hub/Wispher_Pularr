from __future__ import annotations

import unittest
from pathlib import Path

from run_remote_pipeline import _should_skip_sync_path


class RunRemotePipelineTests(unittest.TestCase):
    def test_skips_large_artifacts_and_cache_dirs(self) -> None:
        self.assertTrue(_should_skip_sync_path(Path("downloads/model.safetensors")))
        self.assertTrue(_should_skip_sync_path(Path("project_sync.tar.gz")))
        self.assertTrue(_should_skip_sync_path(Path("reports/metrics.json")))
        self.assertTrue(_should_skip_sync_path(Path("__pycache__/module.pyc")))

    def test_keeps_source_files(self) -> None:
        self.assertFalse(_should_skip_sync_path(Path("train.py")))
        self.assertFalse(_should_skip_sync_path(Path("whisper_pularr/text.py")))


if __name__ == "__main__":
    unittest.main()
