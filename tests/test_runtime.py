from __future__ import annotations

import unittest

from whisper_pularr.runtime import runtime_from_hardware_report


class RuntimeTests(unittest.TestCase):
    def test_hardware_report_t4_prefers_fp16_profile(self) -> None:
        report = {
            "gpu": {
                "name": "Tesla T4",
                "memory_total_gb": 15.0,
                "count": 1,
                "bf16_supported": False,
            },
            "cpu": {"cores": 2},
            "filesystems": [{"mountpoint": "/content", "rotational": 0, "available_gb": 60.0}],
        }

        runtime = runtime_from_hardware_report(report)

        self.assertFalse(runtime.bf16)
        self.assertTrue(runtime.fp16)
        self.assertEqual(runtime.per_device_train_batch_size, 2)
        self.assertEqual(runtime.per_device_eval_batch_size, 4)
        self.assertEqual(runtime.pseudo_label_batch_size, 4)
        self.assertEqual(runtime.profile, "low_vram")


if __name__ == "__main__":
    unittest.main()
