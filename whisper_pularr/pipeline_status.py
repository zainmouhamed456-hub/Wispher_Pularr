from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def select_best_supervised_checkpoint(
    runs_root: Path,
    trial_names: Iterable[str] = ("trial_a", "trial_b", "trial_c"),
) -> str | None:
    candidates: list[tuple[tuple[float, float], str]] = []
    for trial_name in trial_names:
        summary_path = runs_root / trial_name / "run_summary.json"
        checkpoint_path = runs_root / trial_name / "best_full_eval"
        if not summary_path.is_file() or not checkpoint_path.is_dir():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        metrics = summary.get("best_metrics") or {}
        candidates.append(
            (
                (
                    float(metrics.get("normalized_wer", float("inf"))),
                    float(metrics.get("normalized_cer", float("inf"))),
                ),
                str(checkpoint_path),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def sequence_summary_complete(summary_path: Path, manifests_dir: Path) -> bool:
    if not summary_path.is_file():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    manifests = list(manifests_dir.glob("*.jsonl"))
    if not manifests:
        return False
    final_manifest_exists = any(path.stem.endswith("_final") for path in manifests)
    if not final_manifest_exists:
        return False

    for entry in summary.get("runs", []):
        manifest_path = str(entry.get("manifest_path") or "")
        status = str(entry.get("status") or "")
        if manifest_path.endswith("pseudo_labels_final.jsonl") and status in {"completed", "rejected"}:
            return True
    return False
