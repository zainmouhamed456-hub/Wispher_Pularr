from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whisper_pularr.colab_t4_policy import (
    metrics_sort_key,
    resolve_launcher_settings,
    should_promote_checkpoint,
)


DEFAULT_COLAB_SUPERVISED_EPOCHS = "1"
DEFAULT_COLAB_EARLY_STOP_PATIENCE = "1"
DEFAULT_COLAB_MAX_TRAIN_SAMPLES = "1024"
DEFAULT_COLAB_MAX_EVAL_SAMPLES = "64"
DEFAULT_COLAB_PSEUDO_BEAM_SIZE = "3"
DEFAULT_COLAB_PSEUDO_BATCH_SIZE = "2"
DEFAULT_COLAB_MAX_KEEP_MULTIPLE = "0.75"
DEFAULT_COLAB_MANIFEST_EVERY = "4000"
DEFAULT_COLAB_ANALYSIS_TOP_K = "100"


def _configure_runtime_env() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", "/content/hf-cache")
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True, write_through=True)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Colab T4 Whisper Pularr pipeline with promotion-aware defaults.")
    parser.add_argument("root", nargs="?", default="/content/whisper-pularr")
    parser.add_argument("dataset_name", nargs="?", default="google/WaxalNLP")
    parser.add_argument("dataset_config", nargs="?", default="ful_asr")
    parser.add_argument("model_id", nargs="?", default="openai/whisper-small")
    parser.add_argument("teacher_model", nargs="?", default="openai/whisper-large-v3")
    parser.add_argument("whisper_language", nargs="?", default="")
    parser.add_argument("aux_dataset_name", nargs="?", default="")
    parser.add_argument("aux_dataset_config", nargs="?", default="")
    return parser.parse_args()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _log(message: str) -> None:
    print(message, flush=True)


def _run(command: list[str], *, cwd: Path) -> None:
    _log(f"$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=cwd, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _script_path(root: Path, script_name: str) -> str:
    return str(root / script_name)


def _join_beams(beams: tuple[int, ...]) -> str:
    return ",".join(str(beam) for beam in beams)


def _append_if_present(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, value])


def _unique_checkpoints(*checkpoints: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for checkpoint in checkpoints:
        cleaned = str(checkpoint or "").strip()
        if cleaned and cleaned not in seen:
            ordered.append(cleaned)
            seen.add(cleaned)
    return ordered


def _safe_name(checkpoint: str) -> str:
    return checkpoint.replace("\\", "_").replace("/", "_").replace(":", "").replace(".", "_")


def _selected_beam_from_summary(path: Path, fallback_beam: int) -> int:
    if not path.exists():
        return fallback_beam
    payload = _read_json(path)
    return max(int(payload.get("selected_beam") or fallback_beam), 1)


def _ensure_promotion_summary(
    *,
    promotion_summary_path: Path,
    base_checkpoint: str,
    base_metrics: dict[str, Any],
    selected_beam: int,
    full_eval_output_path: str | None,
    comparison_summary_path: Path,
) -> None:
    if promotion_summary_path.exists():
        existing = _read_json(promotion_summary_path)
        checkpoint = str(existing.get("best_checkpoint") or "").strip()
        if checkpoint and Path(checkpoint).exists():
            return
    _save_json(
        promotion_summary_path,
        {
            "best_checkpoint": base_checkpoint,
            "best_metrics": base_metrics,
            "selected_beam": int(selected_beam),
            "full_eval_output_path": full_eval_output_path,
            "comparison_summary_path": str(comparison_summary_path),
            "source": "baseline_eval",
        },
    )


def _promotion_state(promotion_summary_path: Path, fallback_checkpoint: str) -> tuple[str, dict[str, Any] | None]:
    if promotion_summary_path.exists():
        payload = _read_json(promotion_summary_path)
        checkpoint = str(payload.get("best_checkpoint") or "").strip()
        metrics = payload.get("best_metrics")
        if checkpoint:
            return checkpoint, metrics
    return fallback_checkpoint, None


def _supervised_env_template(
    *,
    base_model: str,
    compare_beams: tuple[int, ...],
    fixed_slice_size: int,
    max_train_samples: str,
) -> dict[str, str]:
    return {
        "COLAB_EVAL_ONLY": "0",
        "COLAB_BASE_MODEL": base_model,
        "COLAB_RESUME_FROM": "",
        "COLAB_COMPARE_BEAMS": _join_beams(compare_beams),
        "COLAB_FIXED_SLICE_SIZE": str(fixed_slice_size),
        "COLAB_SUPERVISED_EPOCHS": os.environ.get("COLAB_SUPERVISED_EPOCHS", DEFAULT_COLAB_SUPERVISED_EPOCHS),
        "COLAB_EARLY_STOP_PATIENCE": os.environ.get("COLAB_EARLY_STOP_PATIENCE", DEFAULT_COLAB_EARLY_STOP_PATIENCE),
        "COLAB_MAX_TRAIN_SAMPLES": str(max_train_samples),
        "COLAB_MAX_EVAL_SAMPLES": os.environ.get("COLAB_MAX_EVAL_SAMPLES", DEFAULT_COLAB_MAX_EVAL_SAMPLES),
        "COLAB_RUN_SELF_TRAIN": "0",
    }


def _self_train_env_template(
    *,
    base_model: str,
    compare_beams: tuple[int, ...],
    fixed_slice_size: int,
) -> dict[str, str]:
    return {
        "COLAB_EVAL_ONLY": "0",
        "COLAB_BASE_MODEL": base_model,
        "COLAB_RESUME_FROM": "",
        "COLAB_COMPARE_BEAMS": _join_beams(compare_beams),
        "COLAB_FIXED_SLICE_SIZE": str(fixed_slice_size),
        "COLAB_RUN_SELF_TRAIN": "1",
        "COLAB_PSEUDO_BEAM_SIZE": os.environ.get("COLAB_PSEUDO_BEAM_SIZE", DEFAULT_COLAB_PSEUDO_BEAM_SIZE),
        "COLAB_PSEUDO_BATCH_SIZE": os.environ.get("COLAB_PSEUDO_BATCH_SIZE", DEFAULT_COLAB_PSEUDO_BATCH_SIZE),
        "COLAB_MAX_KEEP_MULTIPLE": os.environ.get("COLAB_MAX_KEEP_MULTIPLE", DEFAULT_COLAB_MAX_KEEP_MULTIPLE),
        "COLAB_MANIFEST_EVERY": os.environ.get("COLAB_MANIFEST_EVERY", DEFAULT_COLAB_MANIFEST_EVERY),
    }


def _compare_checkpoints(
    *,
    root: Path,
    checkpoints: list[str],
    dataset_name: str,
    dataset_config: str,
    whisper_language: str | None,
    output_dir: Path,
    fixed_slice_size: int,
    generation_num_beams: int,
    skip_full: bool,
) -> Path:
    command = [
        sys.executable,
        _script_path(root, "compare_checkpoints.py"),
    ]
    for checkpoint in checkpoints:
        command.extend(["--checkpoint", checkpoint])
    command.extend(
        [
            "--dataset-name",
            dataset_name,
            "--dataset-config",
            dataset_config,
            "--fixed-slice-size",
            str(fixed_slice_size),
            "--generation-num-beams",
            str(generation_num_beams),
            "--output-dir",
            str(output_dir),
        ]
    )
    _append_if_present(command, "--whisper-language", whisper_language)
    if skip_full:
        command.append("--skip-full")
    _run(command, cwd=root)
    return output_dir / "comparison_summary.json"


def _evaluate_checkpoint(
    *,
    root: Path,
    checkpoint: str,
    dataset_name: str,
    dataset_config: str,
    whisper_language: str | None,
    output_path: Path,
    generation_num_beams: int,
) -> Path:
    command = [
        sys.executable,
        _script_path(root, "evaluate.py"),
        "--checkpoint",
        checkpoint,
        "--dataset-name",
        dataset_name,
        "--dataset-config",
        dataset_config,
        "--split",
        "validation",
        "--generation-num-beams",
        str(generation_num_beams),
        "--output-path",
        str(output_path),
    ]
    _append_if_present(command, "--whisper-language", whisper_language)
    _run(command, cwd=root)
    return output_path


def _analyze_eval(root: Path, eval_json: Path, output_path: Path) -> None:
    command = [
        sys.executable,
        _script_path(root, "analyze_eval.py"),
        "--eval-json",
        str(eval_json),
        "--output-path",
        str(output_path),
        "--top-k",
        os.environ.get("COLAB_ANALYSIS_TOP_K", DEFAULT_COLAB_ANALYSIS_TOP_K),
    ]
    _run(command, cwd=root)


def _checkpoint_summary(comparison_summary_path: Path, checkpoint: str) -> dict[str, Any]:
    payload = _read_json(comparison_summary_path)
    for entry in payload.get("checkpoints", []):
        if str(entry.get("checkpoint")) == checkpoint:
            return entry
    raise KeyError(f"Checkpoint summary not found for {checkpoint} in {comparison_summary_path}")


def _run_beam_sweep(
    *,
    root: Path,
    checkpoints: list[str],
    dataset_name: str,
    dataset_config: str,
    whisper_language: str | None,
    reference_checkpoint: str,
    compare_beams: tuple[int, ...],
    fixed_slice_size: int,
    reports_root: Path,
    decode_selection_path: Path,
    promotion_summary_path: Path,
) -> int:
    beam_sweep_root = reports_root / "colab_beam_sweep"
    beam_results: list[dict[str, Any]] = []
    _log(
        "Starting Session 1 beam sweep on the fixed validation slice. "
        f"Beams: {', '.join(str(beam) for beam in compare_beams)}."
    )
    for beam in compare_beams:
        _log(f"Running fixed-slice comparison for beam {beam}...")
        summary_path = _compare_checkpoints(
            root=root,
            checkpoints=checkpoints,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            whisper_language=whisper_language,
            output_dir=beam_sweep_root / f"beam_{beam}",
            fixed_slice_size=fixed_slice_size,
            generation_num_beams=beam,
            skip_full=True,
        )
        checkpoint_summary = _checkpoint_summary(summary_path, reference_checkpoint)
        beam_results.append(
            {
                "beam": int(beam),
                "comparison_summary_path": str(summary_path),
                "reference_checkpoint_metrics": checkpoint_summary["fixed_slice_metrics"],
            }
        )

    beam_results.sort(key=lambda entry: metrics_sort_key(entry["reference_checkpoint_metrics"]) + (entry["beam"],))
    selected_beam = int(beam_results[0]["beam"])
    _log(f"Beam {selected_beam} won the fixed-slice sweep. Running full validation comparison...")
    comparison_summary_path = _compare_checkpoints(
        root=root,
        checkpoints=checkpoints,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        whisper_language=whisper_language,
        output_dir=reports_root / "colab_comparison",
        fixed_slice_size=fixed_slice_size,
        generation_num_beams=selected_beam,
        skip_full=False,
    )
    reference_summary = _checkpoint_summary(comparison_summary_path, reference_checkpoint)
    _save_json(
        decode_selection_path,
        {
            "selected_beam": selected_beam,
            "selection_reference_checkpoint": reference_checkpoint,
            "fixed_slice_size": fixed_slice_size,
            "compare_beams": list(compare_beams),
            "beam_results": beam_results,
            "full_comparison_summary_path": str(comparison_summary_path),
        },
    )
    _ensure_promotion_summary(
        promotion_summary_path=promotion_summary_path,
        base_checkpoint=reference_checkpoint,
        base_metrics=reference_summary.get("full_metrics") or reference_summary["fixed_slice_metrics"],
        selected_beam=selected_beam,
        full_eval_output_path=reference_summary.get("full_output_path"),
        comparison_summary_path=comparison_summary_path,
    )
    full_output_path = str(reference_summary.get("full_output_path") or "").strip()
    if full_output_path:
        _analyze_eval(
            root=root,
            eval_json=Path(full_output_path),
            output_path=reports_root / "colab_base_eval_analysis.json",
        )
    return selected_beam


def _next_session_run_dir(supervised_root: Path) -> Path:
    indices = [
        int(path.name.split("_")[-1])
        for path in supervised_root.glob("session_*")
        if path.is_dir() and path.name.split("_")[-1].isdigit()
    ]
    next_index = (max(indices) + 1) if indices else 1
    return supervised_root / f"session_{next_index:03d}"


def _resume_output_dir(resume_from_checkpoint: str) -> Path:
    checkpoint_path = Path(resume_from_checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path.parent


def _run_supervised_session(
    *,
    root: Path,
    dataset_name: str,
    dataset_config: str,
    model_source: str,
    whisper_language: str | None,
    aux_dataset_name: str | None,
    aux_dataset_config: str | None,
    resume_from_checkpoint: str | None,
    selected_beam: int,
    promotion_summary_path: Path,
    supervised_root: Path,
) -> tuple[Path, Path]:
    run_dir = _resume_output_dir(resume_from_checkpoint) if resume_from_checkpoint else _next_session_run_dir(supervised_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Starting supervised continuation in {run_dir}...")
    command = [
        sys.executable,
        _script_path(root, "train.py"),
        "--stage",
        "supervised",
        "--preset",
        "trial_a",
        "--dataset-name",
        dataset_name,
        "--dataset-config",
        dataset_config,
        "--model-id",
        model_source,
        "--streaming",
        "--num-train-epochs",
        os.environ.get("COLAB_SUPERVISED_EPOCHS", DEFAULT_COLAB_SUPERVISED_EPOCHS),
        "--early-stop-patience-epochs",
        os.environ.get("COLAB_EARLY_STOP_PATIENCE", DEFAULT_COLAB_EARLY_STOP_PATIENCE),
        "--max-train-samples",
        os.environ.get("COLAB_MAX_TRAIN_SAMPLES", DEFAULT_COLAB_MAX_TRAIN_SAMPLES),
        "--max-eval-samples",
        os.environ.get("COLAB_MAX_EVAL_SAMPLES", DEFAULT_COLAB_MAX_EVAL_SAMPLES),
        "--save-total-limit",
        "1",
        "--output-dir",
        str(run_dir),
    ]
    _append_if_present(command, "--whisper-language", whisper_language)
    if aux_dataset_name and aux_dataset_config:
        command.extend(
            [
                "--aux-dataset-name",
                aux_dataset_name,
                "--aux-dataset-config",
                aux_dataset_config,
                "--aux-labeled-repeat-count",
                "1",
            ]
        )
    if resume_from_checkpoint:
        command.extend(["--resume-from-checkpoint", resume_from_checkpoint])
    _run(command, cwd=root)

    candidate_checkpoint = run_dir / "best_full_eval"
    candidate_eval_path = run_dir / "validation_selected_beam.json"
    _evaluate_checkpoint(
        root=root,
        checkpoint=str(candidate_checkpoint),
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        whisper_language=whisper_language,
        output_path=candidate_eval_path,
        generation_num_beams=selected_beam,
    )
    _analyze_eval(
        root=root,
        eval_json=candidate_eval_path,
        output_path=run_dir / "validation_selected_beam.analysis.json",
    )

    current_best_checkpoint, current_best_metrics = _promotion_state(promotion_summary_path, model_source)
    candidate_metrics = _read_json(candidate_eval_path)["metrics"]
    promote, reason = should_promote_checkpoint(candidate_metrics, current_best_metrics)
    decision_payload = {
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_eval_path": str(candidate_eval_path),
        "candidate_metrics": candidate_metrics,
        "comparison_target": current_best_checkpoint,
        "comparison_metrics": current_best_metrics,
        "promote": bool(promote),
        "reason": reason,
    }
    _save_json(run_dir / "promotion_decision.json", decision_payload)
    if promote:
        _save_json(
            promotion_summary_path,
            {
                "best_checkpoint": str(candidate_checkpoint),
                "best_metrics": candidate_metrics,
                "selected_beam": int(selected_beam),
                "full_eval_output_path": str(candidate_eval_path),
                "source": "supervised_session",
                "run_dir": str(run_dir),
                "reason": reason,
            },
        )
    return candidate_checkpoint, candidate_eval_path


def _promote_if_better(
    *,
    promotion_summary_path: Path,
    candidate_checkpoint: str,
    candidate_eval_path: Path,
    selected_beam: int,
    source: str,
    run_dir: Path,
) -> None:
    current_best_checkpoint, current_best_metrics = _promotion_state(promotion_summary_path, candidate_checkpoint)
    candidate_metrics = _read_json(candidate_eval_path)["metrics"]
    promote, reason = should_promote_checkpoint(candidate_metrics, current_best_metrics)
    _save_json(
        run_dir / "promotion_decision.json",
        {
            "candidate_checkpoint": candidate_checkpoint,
            "candidate_eval_path": str(candidate_eval_path),
            "candidate_metrics": candidate_metrics,
            "comparison_target": current_best_checkpoint,
            "comparison_metrics": current_best_metrics,
            "promote": bool(promote),
            "reason": reason,
        },
    )
    if promote:
        _save_json(
            promotion_summary_path,
            {
                "best_checkpoint": candidate_checkpoint,
                "best_metrics": candidate_metrics,
                "selected_beam": int(selected_beam),
                "full_eval_output_path": str(candidate_eval_path),
                "source": source,
                "run_dir": str(run_dir),
                "reason": reason,
            },
        )


def _run_stage_two(
    *,
    root: Path,
    dataset_name: str,
    dataset_config: str,
    model_id: str,
    teacher_model: str,
    whisper_language: str | None,
    selected_beam: int,
    promotion_summary_path: Path,
    artifacts_root: Path,
    reports_root: Path,
    runs_root: Path,
) -> None:
    base_checkpoint, _ = _promotion_state(promotion_summary_path, model_id)
    _log(f"Starting Stage 2 from base checkpoint: {base_checkpoint}")
    pseudo_labels_path = artifacts_root / "colab_pseudo_labels.jsonl"
    pseudo_command = [
        sys.executable,
        _script_path(root, "pseudo_label.py"),
        "--dataset-name",
        dataset_name,
        "--dataset-config",
        dataset_config,
        "--teacher-model",
        teacher_model,
        "--beam-size",
        os.environ.get("COLAB_PSEUDO_BEAM_SIZE", DEFAULT_COLAB_PSEUDO_BEAM_SIZE),
        "--batch-size",
        os.environ.get("COLAB_PSEUDO_BATCH_SIZE", DEFAULT_COLAB_PSEUDO_BATCH_SIZE),
        "--max-keep-multiple",
        os.environ.get("COLAB_MAX_KEEP_MULTIPLE", DEFAULT_COLAB_MAX_KEEP_MULTIPLE),
        "--candidate-pool-multiple",
        "1.0",
        "--manifest-every",
        os.environ.get("COLAB_MANIFEST_EVERY", DEFAULT_COLAB_MANIFEST_EVERY),
        "--output-path",
        str(pseudo_labels_path),
    ]
    _append_if_present(pseudo_command, "--whisper-language", whisper_language)
    _run(pseudo_command, cwd=root)

    self_train_root = runs_root / "colab_self_train"
    sequence_command = [
        sys.executable,
        _script_path(root, "run_self_train_sequence.py"),
        "--dataset-name",
        dataset_name,
        "--dataset-config",
        dataset_config,
        "--model-id",
        model_id,
        "--base-checkpoint",
        base_checkpoint,
        "--manifests-dir",
        str(artifacts_root / "colab_pseudo_labels_manifests"),
        "--output-root",
        str(self_train_root),
        "--final-only",
    ]
    _append_if_present(sequence_command, "--whisper-language", whisper_language)
    _run(sequence_command, cwd=root)

    sequence_summary = _read_json(self_train_root / "sequence_summary.json")
    final_checkpoint = str(sequence_summary["last_best_full_eval_dir"])
    final_eval_path = reports_root / "colab_self_train_validation.json"
    _evaluate_checkpoint(
        root=root,
        checkpoint=final_checkpoint,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        whisper_language=whisper_language,
        output_path=final_eval_path,
        generation_num_beams=selected_beam,
    )
    _analyze_eval(
        root=root,
        eval_json=final_eval_path,
        output_path=reports_root / "colab_self_train_validation.analysis.json",
    )
    _promote_if_better(
        promotion_summary_path=promotion_summary_path,
        candidate_checkpoint=final_checkpoint,
        candidate_eval_path=final_eval_path,
        selected_beam=selected_beam,
        source="self_train",
        run_dir=self_train_root,
    )

    comparison_command = [
        sys.executable,
        _script_path(root, "compare_checkpoints.py"),
        "--checkpoint",
        model_id,
        "--checkpoint",
        base_checkpoint,
        "--checkpoint",
        final_checkpoint,
        "--dataset-name",
        dataset_name,
        "--dataset-config",
        dataset_config,
        "--fixed-slice-size",
        str(max(int(os.environ.get("COLAB_FIXED_SLICE_SIZE") or 64), 1)),
        "--generation-num-beams",
        str(selected_beam),
        "--skip-full",
        "--output-dir",
        str(reports_root / "colab_self_train_comparison"),
    ]
    _append_if_present(comparison_command, "--whisper-language", whisper_language)
    _run(comparison_command, cwd=root)


def main() -> None:
    _configure_runtime_env()
    args = parse_args()
    root = Path(args.root).resolve()
    runs_root = Path(os.environ.get("RUNS_ROOT") or "/content/whisper-pularr-runs").resolve()
    reports_root = runs_root / "reports"
    artifacts_root = runs_root / "artifacts"
    supervised_root = runs_root / "runs" / "colab_supervised"
    promotion_summary_path = reports_root / "colab_promotion_summary.json"
    decode_selection_path = reports_root / "colab_decode_selection.json"

    reports_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    supervised_root.mkdir(parents=True, exist_ok=True)

    settings = resolve_launcher_settings(
        os.environ,
        root=root,
        runs_root=runs_root,
        default_model_id=args.model_id,
        promotion_summary_path=promotion_summary_path,
    )
    whisper_language = str(args.whisper_language or "").strip() or None
    aux_dataset_name = str(args.aux_dataset_name or "").strip() or None
    aux_dataset_config = str(args.aux_dataset_config or "").strip() or None
    checkpoints = _unique_checkpoints(args.model_id, settings.base_model)
    _log(
        "Launcher configuration: "
        f"mode={'eval_only' if settings.eval_only else ('self_train' if os.environ.get('COLAB_RUN_SELF_TRAIN') == '1' else 'supervised')} "
        f"runs_root={runs_root} compare_beams={settings.compare_beams}"
    )

    selected_beam = _selected_beam_from_summary(decode_selection_path, settings.compare_beams[0])
    if settings.eval_only or not decode_selection_path.exists() or not promotion_summary_path.exists():
        selected_beam = _run_beam_sweep(
            root=root,
            checkpoints=checkpoints,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            whisper_language=whisper_language,
            reference_checkpoint=settings.base_model,
            compare_beams=settings.compare_beams,
            fixed_slice_size=settings.fixed_slice_size,
            reports_root=reports_root,
            decode_selection_path=decode_selection_path,
            promotion_summary_path=promotion_summary_path,
        )
    if settings.eval_only:
        promoted_checkpoint, promoted_metrics = _promotion_state(promotion_summary_path, settings.base_model)
        next_step_payload = {
            "phase": "session_2_supervised",
            "recommended_env": _supervised_env_template(
                base_model=promoted_checkpoint,
                compare_beams=settings.compare_beams,
                fixed_slice_size=settings.fixed_slice_size,
                max_train_samples=DEFAULT_COLAB_MAX_TRAIN_SAMPLES,
            ),
            "best_checkpoint": promoted_checkpoint,
            "best_metrics": promoted_metrics,
        }
        _save_json(reports_root / "colab_next_step_session_2.json", next_step_payload)
        print(
            json.dumps(
                {
                    "mode": "eval_only",
                    "selected_beam": selected_beam,
                    "decode_selection_path": str(decode_selection_path),
                    "promotion_summary_path": str(promotion_summary_path),
                    "next_step_path": str(reports_root / "colab_next_step_session_2.json"),
                    "next_step": next_step_payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    candidate_checkpoint, candidate_eval_path = _run_supervised_session(
        root=root,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        model_source=settings.base_model,
        whisper_language=whisper_language,
        aux_dataset_name=aux_dataset_name,
        aux_dataset_config=aux_dataset_config,
        resume_from_checkpoint=settings.resume_from,
        selected_beam=selected_beam,
        promotion_summary_path=promotion_summary_path,
        supervised_root=supervised_root,
    )

    if os.environ.get("COLAB_RUN_SELF_TRAIN") != "1":
        promoted_checkpoint, promoted_metrics = _promotion_state(promotion_summary_path, str(candidate_checkpoint))
        next_step_payload = {
            "phase": "session_3_or_stage_2",
            "recommended_session_3_env": _supervised_env_template(
                base_model=promoted_checkpoint,
                compare_beams=settings.compare_beams,
                fixed_slice_size=settings.fixed_slice_size,
                max_train_samples="4096",
            ),
            "recommended_stage_2_env": _self_train_env_template(
                base_model=promoted_checkpoint,
                compare_beams=settings.compare_beams,
                fixed_slice_size=settings.fixed_slice_size,
            ),
            "best_checkpoint": promoted_checkpoint,
            "best_metrics": promoted_metrics,
        }
        _save_json(candidate_checkpoint.parent / "next_step_recommendations.json", next_step_payload)
        print(
            json.dumps(
                {
                    "mode": "supervised_only",
                    "candidate_checkpoint": str(candidate_checkpoint),
                    "candidate_eval_path": str(candidate_eval_path),
                    "promotion_summary_path": str(promotion_summary_path),
                    "selected_beam": selected_beam,
                    "next_step_path": str(candidate_checkpoint.parent / "next_step_recommendations.json"),
                    "next_step": next_step_payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    _run_stage_two(
        root=root,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        model_id=args.model_id,
        teacher_model=args.teacher_model,
        whisper_language=whisper_language,
        selected_beam=selected_beam,
        promotion_summary_path=promotion_summary_path,
        artifacts_root=artifacts_root,
        reports_root=reports_root,
        runs_root=runs_root / "runs",
    )
    print(
        json.dumps(
            {
                "mode": "self_train_complete",
                "promotion_summary_path": str(promotion_summary_path),
                "selected_beam": selected_beam,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
