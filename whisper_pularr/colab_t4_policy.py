from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_COLAB_COMPARE_BEAMS = (1, 3)
DEFAULT_FIXED_SLICE_SIZE = 64
DEFAULT_PROMOTION_WER_IMPROVEMENT_POINTS = 1.0


@dataclass(frozen=True)
class ColabLauncherSettings:
    eval_only: bool
    base_model: str
    resume_from: str | None
    compare_beams: tuple[int, ...]
    fixed_slice_size: int


def parse_env_flag(value: str | None) -> bool:
    cleaned = str(value or "").strip().lower()
    return cleaned in {"1", "true", "yes", "on"}


def parse_compare_beams(value: str | None) -> tuple[int, ...]:
    if not value:
        return DEFAULT_COLAB_COMPARE_BEAMS

    beams: list[int] = []
    seen: set[int] = set()
    for item in str(value).replace(";", ",").split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        beam = int(cleaned)
        if beam < 1:
            raise ValueError(f"Beam sizes must be >= 1, got: {beam}")
        if beam not in seen:
            beams.append(beam)
            seen.add(beam)
    if not beams:
        raise ValueError("At least one valid beam size is required.")
    return tuple(beams)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_promoted_checkpoint(promotion_summary_path: Path) -> str | None:
    summary = _read_json(promotion_summary_path)
    if not summary:
        return None
    checkpoint = str(summary.get("best_checkpoint") or "").strip()
    if checkpoint and Path(checkpoint).exists():
        return checkpoint
    return None


def resolve_colab_base_model(
    *,
    root: str | Path,
    runs_root: str | Path,
    explicit_base_model: str | None,
    default_model_id: str,
    promotion_summary_path: str | Path | None = None,
) -> str:
    cleaned_explicit = str(explicit_base_model or "").strip()
    if cleaned_explicit:
        return cleaned_explicit

    summary_path = (
        Path(promotion_summary_path)
        if promotion_summary_path is not None
        else Path(runs_root) / "reports" / "colab_promotion_summary.json"
    )
    promoted_checkpoint = _resolve_promoted_checkpoint(summary_path)
    if promoted_checkpoint:
        return promoted_checkpoint

    bundled_checkpoint = Path(root) / "downloads" / "trial_a_best_full_eval" / "best_full_eval"
    if bundled_checkpoint.is_dir():
        return str(bundled_checkpoint)

    return default_model_id


def resolve_launcher_settings(
    env: Mapping[str, str],
    *,
    root: str | Path,
    runs_root: str | Path,
    default_model_id: str,
    promotion_summary_path: str | Path | None = None,
) -> ColabLauncherSettings:
    compare_beams = parse_compare_beams(env.get("COLAB_COMPARE_BEAMS"))
    fixed_slice_size = max(int(env.get("COLAB_FIXED_SLICE_SIZE") or DEFAULT_FIXED_SLICE_SIZE), 1)
    base_model = resolve_colab_base_model(
        root=root,
        runs_root=runs_root,
        explicit_base_model=env.get("COLAB_BASE_MODEL"),
        default_model_id=default_model_id,
        promotion_summary_path=promotion_summary_path,
    )
    resume_from = str(env.get("COLAB_RESUME_FROM") or "").strip() or None
    return ColabLauncherSettings(
        eval_only=parse_env_flag(env.get("COLAB_EVAL_ONLY")),
        base_model=base_model,
        resume_from=resume_from,
        compare_beams=compare_beams,
        fixed_slice_size=fixed_slice_size,
    )


def metrics_sort_key(metrics: Mapping[str, Any] | None) -> tuple[float, float]:
    metrics = metrics or {}
    return (
        float(metrics.get("normalized_wer", float("inf"))),
        float(metrics.get("normalized_cer", float("inf"))),
    )


def should_promote_checkpoint(
    candidate_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any] | None,
    *,
    min_wer_improvement_points: float = DEFAULT_PROMOTION_WER_IMPROVEMENT_POINTS,
) -> tuple[bool, str]:
    if reference_metrics is None:
        return True, "no_reference_metrics"

    candidate_wer = float(candidate_metrics["normalized_wer"])
    candidate_cer = float(candidate_metrics["normalized_cer"])
    reference_wer = float(reference_metrics["normalized_wer"])
    reference_cer = float(reference_metrics["normalized_cer"])
    required_delta = float(min_wer_improvement_points) / 100.0
    epsilon = 1e-9

    if candidate_wer <= reference_wer - required_delta + epsilon:
        return True, "wer_improved"
    if abs(candidate_wer - reference_wer) <= epsilon and candidate_cer < reference_cer - epsilon:
        return True, "wer_tied_cer_improved"
    return False, "did_not_meet_promotion_threshold"
