from __future__ import annotations

from typing import Any


def metrics_key(metrics: dict[str, Any]) -> tuple[float, float]:
    return (float(metrics["normalized_wer"]), float(metrics["normalized_cer"]))


def beats_reference(candidate_metrics: dict[str, Any], reference_metrics: dict[str, Any]) -> bool:
    return metrics_key(candidate_metrics) < metrics_key(reference_metrics)
