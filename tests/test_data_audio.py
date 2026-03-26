from __future__ import annotations

import math
import unittest
from unittest import mock

from whisper_pularr import data


class _FailingTorchaudio:
    @staticmethod
    def info(_source):
        raise RuntimeError("Failed to decode audio.")


class DataAudioTests(unittest.TestCase):
    def test_audio_duration_returns_infinity_for_undecodable_bytes(self) -> None:
        with mock.patch.object(data, "torchaudio", _FailingTorchaudio):
            duration = data.audio_duration_seconds({"audio": {"bytes": b"broken"}})
        self.assertTrue(math.isinf(duration))


if __name__ == "__main__":
    unittest.main()
