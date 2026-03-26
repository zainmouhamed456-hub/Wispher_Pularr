from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from whisper_pularr.sequence_policy import beats_reference


LAUNCH_METADATA_FILENAME = "sequence_launch.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-training sequentially across pseudo-label manifest snapshots.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--whisper-language", default=None)
    parser.add_argument("--hardware-report", default=None)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--manifests-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--max-manifests", type=int, default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=10)
    return parser.parse_args()


def _manifest_sort_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    if stem.endswith("_final"):
        return (1, "final")
    suffix = stem.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return (0, int(suffix))
    return (0, stem)


def _iter_manifests(
    manifests_dir: Path,
    *,
    final_only: bool = False,
    max_manifests: int | None = None,
) -> list[Path]:
    manifests = sorted(manifests_dir.glob("*.jsonl"), key=_manifest_sort_key)
    if final_only:
        final_manifests = [path for path in manifests if path.stem.endswith("_final")]
        return final_manifests[-1:] if final_manifests else []
    if max_manifests and max_manifests > 0:
        return manifests[-int(max_manifests) :]
    return manifests


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _metrics_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload.get("best_metrics") or payload.get("metrics") or {}
    normalized_wer = metrics.get("normalized_wer")
    normalized_cer = metrics.get("normalized_cer")
    if normalized_wer is None or normalized_cer is None:
        raise ValueError(f"Missing normalized metrics in payload: {payload}")
    return {
        "normalized_wer": float(normalized_wer),
        "normalized_cer": float(normalized_cer),
        "raw_wer": float(metrics.get("raw_wer", float("inf"))),
        "raw_cer": float(metrics.get("raw_cer", float("inf"))),
    }


def _resolve_baseline_summary(base_checkpoint: str | None) -> tuple[str | None, dict[str, float] | None]:
    if not base_checkpoint:
        return None, None
    checkpoint_path = Path(base_checkpoint)
    run_summary_path = checkpoint_path.parent / "run_summary.json"
    if run_summary_path.exists():
        summary = _read_json(run_summary_path)
        return str(checkpoint_path), _metrics_from_payload(summary)
    best_summary_path = checkpoint_path.parent / "best_full_eval_summary.json"
    if best_summary_path.exists():
        summary = _read_json(best_summary_path)
        return str(checkpoint_path), {
            "normalized_wer": float(summary["best_normalized_wer"]) / 100.0,
            "normalized_cer": float(summary["best_normalized_cer"]) / 100.0,
            "raw_wer": float("inf"),
            "raw_cer": float("inf"),
        }
    return str(checkpoint_path), None


def _launch_metadata_path(run_dir: Path) -> Path:
    return run_dir / LAUNCH_METADATA_FILENAME


def _write_launch_metadata(run_dir: Path, payload: dict[str, Any]) -> None:
    _save_json(_launch_metadata_path(run_dir), payload)


def _read_launch_metadata(run_dir: Path) -> dict[str, Any]:
    path = _launch_metadata_path(run_dir)
    if not path.exists():
        return {}
    return _read_json(path)


def _build_pending_entry(manifest_path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "run_dir": str(run_dir),
        "run_summary_path": str(run_dir / "run_summary.json"),
        "best_model_dir": str(run_dir / "best_full_eval"),
        "best_epoch": None,
        "normalized_wer": None,
        "normalized_cer": None,
        "raw_wer": None,
        "raw_cer": None,
        "status": "pending",
        "base_checkpoint_used": None,
        "promoted": False,
        "comparison_target": None,
        "rejection_reason": None,
    }


def _build_sequence_summary(
    manifests_dir: Path,
    manifests: list[Path],
    output_root: Path,
    *,
    baseline_checkpoint: str | None,
    baseline_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    best_checkpoint = baseline_checkpoint
    best_metrics = baseline_metrics
    last_promoted_run_dir: str | None = None
    rejected_manifest_paths: list[str] = []

    for manifest_path in manifests:
        run_dir = output_root / manifest_path.stem
        run_summary_path = run_dir / "run_summary.json"
        launch_metadata = _read_launch_metadata(run_dir)
        entry = _build_pending_entry(manifest_path, run_dir)
        entry["base_checkpoint_used"] = launch_metadata.get("base_checkpoint_used") or best_checkpoint
        entry["comparison_target"] = launch_metadata.get("comparison_target") or best_checkpoint

        if run_summary_path.exists():
            run_summary = _read_json(run_summary_path)
            run_metrics = _metrics_from_payload(run_summary)
            entry.update(
                {
                    "best_model_dir": str(run_summary.get("best_model_dir") or (run_dir / "best_full_eval")),
                    "best_epoch": run_summary.get("best_epoch"),
                    "normalized_wer": run_metrics["normalized_wer"],
                    "normalized_cer": run_metrics["normalized_cer"],
                    "raw_wer": run_metrics["raw_wer"],
                    "raw_cer": run_metrics["raw_cer"],
                }
            )
            comparison_target = entry["comparison_target"] or best_checkpoint
            entry["comparison_target"] = comparison_target
            if best_metrics is None or beats_reference(run_metrics, best_metrics):
                entry["status"] = "completed"
                entry["promoted"] = True
                best_checkpoint = entry["best_model_dir"]
                best_metrics = run_metrics
                last_promoted_run_dir = str(run_dir)
            else:
                entry["status"] = "rejected"
                entry["promoted"] = False
                entry["rejection_reason"] = "did_not_beat_best_so_far"
                rejected_manifest_paths.append(str(manifest_path))
        elif run_dir.exists():
            entry["status"] = "running"
        runs.append(entry)

    return {
        "manifests_dir": str(manifests_dir),
        "output_root": str(output_root),
        "baseline_checkpoint": baseline_checkpoint,
        "baseline_metrics": baseline_metrics,
        "runs": runs,
        "last_run_dir": last_promoted_run_dir,
        "last_best_full_eval_dir": best_checkpoint,
        "stop_requested": False,
        "rejected_manifest_path": rejected_manifest_paths[-1] if rejected_manifest_paths else None,
        "rejected_manifest_paths": rejected_manifest_paths,
    }


def _next_launch_candidate(summary: dict[str, Any]) -> dict[str, Any] | None:
    for entry in summary.get("runs", []):
        if entry.get("status") == "pending":
            return entry
    return None


def _launch_self_train(
    args: argparse.Namespace,
    *,
    manifest_path: str,
    run_dir: str,
    base_checkpoint: str | None,
    comparison_target: str | None,
) -> None:
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    _write_launch_metadata(
        run_dir_path,
        {
            "manifest_path": manifest_path,
            "base_checkpoint_used": base_checkpoint,
            "comparison_target": comparison_target,
        },
    )

    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train.py"),
        "--stage",
        "self_train",
        "--preset",
        "trial_b",
        "--dataset-name",
        args.dataset_name,
        "--dataset-config",
        args.dataset_config,
        "--model-id",
        args.model_id,
        "--pseudo-labels-path",
        manifest_path,
        "--output-dir",
        run_dir,
    ]
    if args.whisper_language:
        command.extend(["--whisper-language", args.whisper_language])
    if args.hardware_report:
        command.extend(["--hardware-report", args.hardware_report])
    if base_checkpoint:
        command.extend(["--base-checkpoint", base_checkpoint])
    _run_command(command)


def main() -> None:
    args = parse_args()
    manifests_dir = Path(args.manifests_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_checkpoint, baseline_metrics = _resolve_baseline_summary(args.base_checkpoint)

    while True:
        manifests = _iter_manifests(
            manifests_dir,
            final_only=bool(args.final_only),
            max_manifests=args.max_manifests,
        )
        if not manifests and not args.watch:
            raise SystemExit(f"No manifest snapshots found in {manifests_dir}")

        summary = _build_sequence_summary(
            manifests_dir,
            manifests,
            output_root,
            baseline_checkpoint=baseline_checkpoint,
            baseline_metrics=baseline_metrics,
        )
        _save_json(output_root / "sequence_summary.json", summary)

        next_entry = _next_launch_candidate(summary)
        if next_entry is not None:
            _launch_self_train(
                args,
                manifest_path=str(next_entry["manifest_path"]),
                run_dir=str(next_entry["run_dir"]),
                base_checkpoint=next_entry.get("base_checkpoint_used"),
                comparison_target=next_entry.get("comparison_target"),
            )
            continue

        final_manifest_exists = any(path.stem.endswith("_final") for path in manifests)
        final_run_recorded = any(
            entry.get("manifest_path", "").endswith("pseudo_labels_final.jsonl")
            and entry.get("status") in {"completed", "rejected"}
            for entry in summary.get("runs", [])
        )
        if not args.watch:
            break
        if final_manifest_exists and final_run_recorded:
            break
        time.sleep(max(int(args.poll_seconds), 1))

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
