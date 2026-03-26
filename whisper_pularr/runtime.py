from __future__ import annotations

import os
from importlib.util import find_spec
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RuntimeConfig:
    bf16: bool
    fp16: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    pseudo_label_batch_size: int
    evaluation_batch_size: int
    generation_num_beams: int
    save_steps: int
    logging_steps: int
    use_multi_gpu: bool
    cache_root: str
    output_root: str
    report_path: str | None = None
    vram_gb: float | None = None
    accelerator_name: str | None = None
    profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pick_fast_mount(report: dict[str, Any]) -> str:
    filesystems = report.get("filesystems", [])
    ranked = sorted(
        filesystems,
        key=lambda fs: (
            0 if fs.get("rotational") == 0 else 1,
            -float(fs.get("available_gb", 0.0)),
        ),
    )
    for fs in ranked:
        mountpoint = fs.get("mountpoint")
        if mountpoint and mountpoint not in {"/boot", "/snap"}:
            return mountpoint
    return str(Path.home())


def _colab_output_root() -> Path | None:
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        return drive_root / "whisper-pularr-runs"
    return None


def _build_runtime_config(
    *,
    vram_gb: float,
    gpu_count: int,
    physical_cores: int,
    bf16_supported: bool,
    chosen_root: str,
    report_path: str | None = None,
    accelerator_name: str | None = None,
    is_colab: bool = False,
) -> RuntimeConfig:
    profile = "cpu"
    if vram_gb >= 70:
        train_batch_size = 24
        gradient_accumulation = 2
        eval_batch_size = 24
        pseudo_batch_size = 32
        save_steps = 1000
        logging_steps = 20
        profile = "ultra_vram"
    elif vram_gb >= 40:
        train_batch_size = 8
        gradient_accumulation = 8
        eval_batch_size = 12
        pseudo_batch_size = 20
        save_steps = 500
        logging_steps = 25
        profile = "high_vram"
    elif vram_gb >= 20:
        train_batch_size = 4
        gradient_accumulation = 16
        eval_batch_size = 8
        pseudo_batch_size = 12
        save_steps = 250
        logging_steps = 25
        profile = "mid_vram"
    elif vram_gb >= 14:
        train_batch_size = 2
        gradient_accumulation = 16
        eval_batch_size = 4
        pseudo_batch_size = 4
        save_steps = 125
        logging_steps = 10
        profile = "colab_t4" if is_colab else "low_vram"
    else:
        train_batch_size = 1
        gradient_accumulation = 24 if vram_gb > 0 else 8
        eval_batch_size = 2
        pseudo_batch_size = 2
        save_steps = 100
        logging_steps = 10
        profile = "very_low_vram" if vram_gb > 0 else "cpu"

    if is_colab:
        num_workers = 2
        prefetch_factor = 2
    else:
        num_workers = max(2, min(physical_cores // max(gpu_count, 1), 12))
        prefetch_factor = 4 if num_workers >= 8 else 2

    cache_root = str(Path(chosen_root) / "hf-cache")
    output_root_path = _colab_output_root() if is_colab else None
    if output_root_path is None:
        output_root_path = Path(chosen_root) / "whisper-pularr-runs"

    return RuntimeConfig(
        bf16=bool(bf16_supported and vram_gb >= 20),
        fp16=bool(vram_gb > 0 and not (bf16_supported and vram_gb >= 20)),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        dataloader_num_workers=num_workers,
        dataloader_prefetch_factor=prefetch_factor,
        pseudo_label_batch_size=pseudo_batch_size,
        evaluation_batch_size=eval_batch_size,
        generation_num_beams=5,
        save_steps=save_steps,
        logging_steps=logging_steps,
        use_multi_gpu=gpu_count > 1,
        cache_root=cache_root,
        output_root=str(output_root_path),
        report_path=report_path,
        vram_gb=vram_gb,
        accelerator_name=accelerator_name,
        profile=profile,
    )


def runtime_from_hardware_report(report: dict[str, Any], output_root: str | None = None) -> RuntimeConfig:
    gpu = report.get("gpu", {})
    vram_gb = float(gpu.get("memory_total_gb", 0.0))
    gpu_count = int(gpu.get("count", 1) or 1)
    cpu = report.get("cpu", {})
    physical_cores = int(cpu.get("cores", 8) or 8)
    chosen_root = output_root or _pick_fast_mount(report)
    return _build_runtime_config(
        vram_gb=vram_gb,
        gpu_count=gpu_count,
        physical_cores=physical_cores,
        bf16_supported=bool(gpu.get("bf16_supported", vram_gb >= 40)),
        chosen_root=chosen_root,
        report_path=report.get("report_path"),
        accelerator_name=str(gpu.get("name") or "") or None,
    )


def _detect_local_runtime() -> RuntimeConfig:
    is_colab = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))
    if not is_colab:
        is_colab = find_spec("google.colab") is not None or Path("/content").exists()

    chosen_root = "/content" if is_colab and Path("/content").exists() else str(Path.home())
    physical_cores = os.cpu_count() or 8
    vram_gb = 0.0
    gpu_count = 1
    bf16_supported = False
    accelerator_name: str | None = None

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = int(torch.cuda.device_count() or 1)
            properties = torch.cuda.get_device_properties(0)
            vram_gb = float(properties.total_memory / (1024**3))
            accelerator_name = str(getattr(properties, "name", "") or "") or None
            bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        pass

    return _build_runtime_config(
        vram_gb=vram_gb,
        gpu_count=gpu_count,
        physical_cores=physical_cores,
        bf16_supported=bf16_supported,
        chosen_root=chosen_root,
        accelerator_name=accelerator_name,
        is_colab=is_colab,
    )


def runtime_from_optional_report(report_path: str | None) -> RuntimeConfig:
    if not report_path:
        return _detect_local_runtime()

    import json

    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if "hardware_report" in report:
        report = report["hardware_report"]
    report["report_path"] = report_path
    return runtime_from_hardware_report(report)
