#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path


def run(*command: str) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.stdout.strip()


def parse_gpu() -> dict:
    gpu_info = {
        "count": 0,
        "name": None,
        "memory_total_gb": 0.0,
        "driver_version": None,
        "cuda_version": None,
        "bf16_supported": False,
        "gpus": [],
    }
    if not shutil.which("nvidia-smi"):
        return gpu_info
    lines = run(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,cuda_version",
        "--format=csv,noheader,nounits",
    ).splitlines()
    for line in lines:
        if not line.strip():
            continue
        name, memory_mb, driver_version, cuda_version = [part.strip() for part in line.split(",")]
        gpu_info["gpus"].append(
            {
                "name": name,
                "memory_total_gb": round(float(memory_mb) / 1024.0, 2),
                "driver_version": driver_version,
                "cuda_version": cuda_version,
            }
        )
    if gpu_info["gpus"]:
        gpu_info["count"] = len(gpu_info["gpus"])
        gpu_info["name"] = gpu_info["gpus"][0]["name"]
        gpu_info["memory_total_gb"] = gpu_info["gpus"][0]["memory_total_gb"]
        gpu_info["driver_version"] = gpu_info["gpus"][0]["driver_version"]
        gpu_info["cuda_version"] = gpu_info["gpus"][0]["cuda_version"]
        bf16_names = ("H100", "H200", "A100", "L40", "L40S", "RTX 6000 Ada")
        gpu_info["bf16_supported"] = any(token.lower() in gpu_info["name"].lower() for token in bf16_names)
    return gpu_info


def parse_cpu() -> dict:
    logical = int(run("nproc") or "0")
    cores = logical
    sockets = 1
    lscpu_json = run("lscpu", "-J")
    if lscpu_json:
        try:
            parsed = json.loads(lscpu_json)
            rows = parsed.get("lscpu", [])
            mapping = {row["field"].rstrip(":"): row["data"] for row in rows}
            cores = int(mapping.get("Core(s) per socket", cores))
            sockets = int(mapping.get("Socket(s)", sockets))
        except Exception:
            pass
    return {
        "logical_cpus": logical,
        "cores": cores * sockets,
        "sockets": sockets,
        "arch": platform.machine(),
    }


def parse_memory() -> dict:
    total_bytes = 0
    available_bytes = 0
    free_output = run("free", "-b")
    for line in free_output.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            total_bytes = int(parts[1])
            available_bytes = int(parts[6]) if len(parts) > 6 else int(parts[3])
            break
    return {
        "total_gb": round(total_bytes / (1024 ** 3), 2),
        "available_gb": round(available_bytes / (1024 ** 3), 2),
    }


def parse_filesystems() -> list[dict]:
    filesystems = []
    lsblk_json = run("lsblk", "-J", "-o", "NAME,ROTA,SIZE,TYPE,MOUNTPOINT")
    block_info = {}
    if lsblk_json:
        try:
            parsed = json.loads(lsblk_json)

            def walk(devices):
                for device in devices:
                    mountpoint = device.get("mountpoint")
                    if mountpoint:
                        block_info[mountpoint] = {"name": device.get("name"), "rotational": device.get("rota")}
                    walk(device.get("children") or [])

            walk(parsed.get("blockdevices", []))
        except Exception:
            pass

    for mountpoint in {"/", "/home", "/mnt", "/workspace"}:
        if not Path(mountpoint).exists():
            continue
        usage = shutil.disk_usage(mountpoint)
        details = block_info.get(mountpoint, {})
        filesystems.append(
            {
                "mountpoint": mountpoint,
                "available_gb": round(usage.free / (1024 ** 3), 2),
                "total_gb": round(usage.total / (1024 ** 3), 2),
                "device": details.get("name"),
                "rotational": details.get("rotational"),
            }
        )
    return filesystems


def measure_network() -> dict | None:
    if os.environ.get("AUDIT_NETWORK", "0") != "1":
        return None
    import urllib.request

    start = time.time()
    try:
        with urllib.request.urlopen("https://huggingface.co", timeout=10) as response:
            response.read(64)
        elapsed = time.time() - start
        return {"huggingface_head_seconds": round(elapsed, 3)}
    except Exception as exc:
        return {"error": str(exc)}


payload = {
    "hostname": platform.node(),
    "kernel": platform.platform(),
    "timestamp_epoch": int(time.time()),
    "gpu": parse_gpu(),
    "cpu": parse_cpu(),
    "memory": parse_memory(),
    "filesystems": parse_filesystems(),
    "network": measure_network(),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
