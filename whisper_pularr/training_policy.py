from __future__ import annotations

from dataclasses import replace as dataclass_replace
from typing import Any

from .settings import SUPERVISED_PRESETS


def runtime_for_stage(stage: str, runtime: Any, use_cuda: bool) -> Any:
    if stage != "self_train" or not use_cuda:
        return runtime
    tuned = dataclass_replace(runtime)
    vram_gb = float(getattr(runtime, "vram_gb", 0.0) or 0.0)
    if vram_gb >= 70:
        tuned.per_device_train_batch_size = max(int(runtime.per_device_train_batch_size), 32)
        tuned.gradient_accumulation_steps = 2 if int(runtime.gradient_accumulation_steps) > 1 else 1
        tuned.per_device_eval_batch_size = max(int(runtime.per_device_eval_batch_size), 32)
        tuned.evaluation_batch_size = max(int(runtime.evaluation_batch_size), 32)
        tuned.dataloader_num_workers = max(int(runtime.dataloader_num_workers), 16)
        tuned.dataloader_prefetch_factor = max(int(runtime.dataloader_prefetch_factor), 4)
    elif vram_gb >= 40:
        tuned.per_device_train_batch_size = max(int(runtime.per_device_train_batch_size), 8)
        tuned.gradient_accumulation_steps = min(max(int(runtime.gradient_accumulation_steps), 4), 8)
        tuned.per_device_eval_batch_size = max(int(runtime.per_device_eval_batch_size), 12)
        tuned.evaluation_batch_size = max(int(runtime.evaluation_batch_size), 12)
        tuned.dataloader_num_workers = max(int(runtime.dataloader_num_workers), 8)
        tuned.dataloader_prefetch_factor = max(int(runtime.dataloader_prefetch_factor), 2)
    elif vram_gb >= 14:
        tuned.per_device_train_batch_size = min(max(int(runtime.per_device_train_batch_size), 2), 2)
        tuned.gradient_accumulation_steps = max(int(runtime.gradient_accumulation_steps), 16)
        tuned.per_device_eval_batch_size = min(max(int(runtime.per_device_eval_batch_size), 4), 4)
        tuned.evaluation_batch_size = min(max(int(runtime.evaluation_batch_size), 4), 4)
        tuned.dataloader_num_workers = min(max(int(runtime.dataloader_num_workers), 2), 4)
        tuned.dataloader_prefetch_factor = min(max(int(runtime.dataloader_prefetch_factor), 2), 2)
    else:
        tuned.per_device_train_batch_size = 1
        tuned.gradient_accumulation_steps = max(int(runtime.gradient_accumulation_steps), 24)
        tuned.per_device_eval_batch_size = min(max(int(runtime.per_device_eval_batch_size), 2), 2)
        tuned.evaluation_batch_size = min(max(int(runtime.evaluation_batch_size), 2), 2)
        tuned.dataloader_num_workers = min(max(int(runtime.dataloader_num_workers), 1), 2)
        tuned.dataloader_prefetch_factor = 2
    return tuned


def resolve_label_smoothing_factor(preset: str) -> float:
    return float(SUPERVISED_PRESETS[preset]["label_smoothing_factor"])


def applied_label_smoothing_factor(*, stage: str, runtime_profile: str | None, requested: float) -> float:
    if stage == "supervised" and runtime_profile == "colab_t4":
        return 0.0
    return float(requested)
