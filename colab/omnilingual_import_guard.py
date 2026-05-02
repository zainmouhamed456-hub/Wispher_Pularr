from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = Path("/content/drive/MyDrive/omnilingual-pularr-runs")
DEFAULT_HF_HOME = Path("/content/hf-cache")


def _external_paths(runs_root: str | Path) -> list[Path]:
    external_root = Path(runs_root) / "external" / "omnilingual-asr"
    return [external_root / "src", external_root]


def _add_external_paths(runs_root: str | Path) -> None:
    for path in reversed(_external_paths(runs_root)):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _import_pipeline() -> tuple[Any, Any]:
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    return torch, ASRInferencePipeline


def ensure_omnilingual_pipeline(
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    hf_home: str | Path = DEFAULT_HF_HOME,
    force_bootstrap: bool = False,
) -> tuple[Any, Any]:
    runs_root = Path(runs_root)
    hf_home = Path(hf_home)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    runs_root.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    _add_external_paths(runs_root)
    if not force_bootstrap:
        try:
            return _import_pipeline()
        except ModuleNotFoundError:
            pass

    subprocess.check_call(
        [
            sys.executable,
            str(REPO_ROOT / "colab" / "run_omnilingual_t4_free.py"),
            "bootstrap",
            "--runs-root",
            str(runs_root),
            "--hf-cache",
            str(hf_home),
        ],
        cwd=REPO_ROOT,
    )
    _add_external_paths(runs_root)
    importlib.invalidate_caches()
    return _import_pipeline()


def print_omnilingual_status(
    *,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    hf_home: str | Path = DEFAULT_HF_HOME,
    force_bootstrap: bool = False,
) -> tuple[Any, Any]:
    torch, ASRInferencePipeline = ensure_omnilingual_pipeline(
        runs_root=runs_root,
        hf_home=hf_home,
        force_bootstrap=force_bootstrap,
    )
    print("cuda_available=", torch.cuda.is_available())
    print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    print("pipeline=", ASRInferencePipeline.__name__)
    return torch, ASRInferencePipeline


if __name__ == "__main__":
    print_omnilingual_status()
