from __future__ import annotations

import unittest

from whisper_pularr.runtime import RuntimeConfig
from whisper_pularr.training_policy import (
    applied_label_smoothing_factor,
    resolve_label_smoothing_factor,
    runtime_for_stage,
)


class TrainingPolicyTests(unittest.TestCase):
    def test_runtime_for_self_train_keeps_generation_beams(self) -> None:
        runtime = RuntimeConfig(
            bf16=True,
            fp16=False,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            dataloader_num_workers=8,
            dataloader_prefetch_factor=2,
            pseudo_label_batch_size=8,
            evaluation_batch_size=4,
            generation_num_beams=5,
            save_steps=100,
            logging_steps=10,
            use_multi_gpu=False,
            cache_root="cache",
            output_root="out",
            vram_gb=80.0,
        )

        tuned = runtime_for_stage("self_train", runtime, use_cuda=True)

        self.assertEqual(tuned.generation_num_beams, 5)
        self.assertGreaterEqual(tuned.per_device_train_batch_size, 32)

    def test_runtime_for_self_train_stays_conservative_on_t4(self) -> None:
        runtime = RuntimeConfig(
            bf16=False,
            fp16=True,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=16,
            dataloader_num_workers=2,
            dataloader_prefetch_factor=2,
            pseudo_label_batch_size=4,
            evaluation_batch_size=4,
            generation_num_beams=5,
            save_steps=100,
            logging_steps=10,
            use_multi_gpu=False,
            cache_root="cache",
            output_root="out",
            vram_gb=15.0,
            accelerator_name="Tesla T4",
            profile="colab_t4",
        )

        tuned = runtime_for_stage("self_train", runtime, use_cuda=True)

        self.assertEqual(tuned.generation_num_beams, 5)
        self.assertEqual(tuned.per_device_train_batch_size, 2)
        self.assertEqual(tuned.per_device_eval_batch_size, 4)
        self.assertEqual(tuned.evaluation_batch_size, 4)

    def test_label_smoothing_comes_from_preset(self) -> None:
        self.assertEqual(resolve_label_smoothing_factor("trial_a"), 0.1)
        self.assertEqual(resolve_label_smoothing_factor("trial_c"), 0.0)

    def test_colab_supervised_disables_label_smoothing(self) -> None:
        self.assertEqual(
            applied_label_smoothing_factor(stage="supervised", runtime_profile="colab_t4", requested=0.1),
            0.0,
        )
        self.assertEqual(
            applied_label_smoothing_factor(stage="supervised", runtime_profile="low_vram", requested=0.1),
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
