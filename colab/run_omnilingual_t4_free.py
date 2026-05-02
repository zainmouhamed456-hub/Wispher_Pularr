from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whisper_pularr.omnilingual_colab import (
    DATASET_CARD_NAME,
    DEFAULT_OMNI_BASELINE_MODELS,
    DEFAULT_OMNI_BASELINE_SAMPLES,
    DEFAULT_OMNI_LANG,
    DEFAULT_OMNI_MAIN_STEPS,
    DEFAULT_OMNI_MAX_DURATION_SECONDS,
    DEFAULT_OMNI_SMOKE_STEPS,
    collect_ref_hyp_metrics,
    copy_dataset_card_to_external_repo,
    ensure_external_omnilingual_repo,
    install_colab_omnilingual_dependencies,
    env_with_external_omnilingual_paths,
    prepare_waxal_omnilingual_dataset,
    read_json,
    run_command,
    run_official_train_recipe,
    save_json,
    select_baseline_model,
    should_promote_omnilingual,
    verify_colab_omnilingual_import,
    write_omnilingual_dataset_assets,
)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    return max(int(value), 1)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Omnilingual ASR Pularr fine-tuning on Google Colab T4.")
    parser.add_argument("mode", choices=["bootstrap", "baseline", "prepare", "train", "eval", "promote", "all"])
    parser.add_argument("--dataset-name", default="google/WaxalNLP")
    parser.add_argument("--dataset-config", default="ful_asr")
    parser.add_argument("--lang", default=os.environ.get("OMNI_LANG", DEFAULT_OMNI_LANG))
    parser.add_argument("--runs-root", default=os.environ.get("RUNS_ROOT") or "/content/drive/MyDrive/omnilingual-pularr-runs")
    parser.add_argument("--hf-cache", default=os.environ.get("HF_HOME") or "/content/hf-cache")
    parser.add_argument("--baseline-model", action="append", dest="baseline_models")
    parser.add_argument("--baseline-samples", type=int, default=_env_int("OMNI_BASELINE_SAMPLES", DEFAULT_OMNI_BASELINE_SAMPLES))
    parser.add_argument("--max-duration-seconds", type=float, default=_env_float("OMNI_MAX_DURATION_SECONDS", DEFAULT_OMNI_MAX_DURATION_SECONDS))
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--main-steps", type=int, default=_env_int("OMNI_MAIN_STEPS", DEFAULT_OMNI_MAIN_STEPS))
    parser.add_argument("--smoke-steps", type=int, default=_env_int("OMNI_SMOKE_STEPS", DEFAULT_OMNI_SMOKE_STEPS))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("OMNI_LEARNING_RATE") or 1e-5))
    parser.add_argument("--grad-accumulation", type=int, default=_env_int("OMNI_GRAD_ACCUMULATION", 16))
    parser.add_argument("--validate-every", type=int, default=_env_int("OMNI_VALIDATE_EVERY", 250))
    parser.add_argument("--checkpoint-every", type=int, default=_env_int("OMNI_CHECKPOINT_EVERY", 500))
    parser.add_argument("--install-deps", action="store_true", help="Clone/install Meta Omnilingual and CUDA dependencies before running.")
    parser.add_argument("--skip-train", action="store_true", help="Generate train configs but do not launch the official recipe.")
    parser.add_argument("--eval-dir", default=None, help="Official eval output directory containing transcriptions/.")
    parser.add_argument("--candidate-checkpoint", default=None, help="Candidate checkpoint path/card for promotion.")
    parser.add_argument("--candidate-eval", default=None, help="Candidate eval JSON for promotion.")
    return parser.parse_args()


def _configure_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HOME", str(args.hf_cache))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(args.hf_cache).mkdir(parents=True, exist_ok=True)
    Path(args.runs_root).mkdir(parents=True, exist_ok=True)


def _maybe_install_dependencies(args: argparse.Namespace) -> None:
    if not args.install_deps:
        return
    external_root = ensure_external_omnilingual_repo(runs_root=args.runs_root)
    install_colab_omnilingual_dependencies(external_root)


def _selected_model_or_default(runs_root: Path) -> str:
    summary_path = runs_root / "reports" / "omnilingual_baseline_summary.json"
    if summary_path.exists():
        selected = str(read_json(summary_path).get("selected_model_card") or "").strip()
        if selected:
            return selected
    return DEFAULT_OMNI_BASELINE_MODELS[0]


def _prepare_configs(args: argparse.Namespace, *, steps: int) -> tuple[Path, Path, Path]:
    base_model = _selected_model_or_default(Path(args.runs_root))
    max_audio_len = int(float(args.max_duration_seconds) * 16000)
    return write_omnilingual_dataset_assets(
        runs_root=args.runs_root,
        lang=args.lang,
        base_model=base_model,
        max_audio_len=max_audio_len,
        max_num_elements=max_audio_len,
        grad_accumulation=args.grad_accumulation,
        learning_rate=args.learning_rate,
        num_steps=steps,
        validate_every=args.validate_every,
        checkpoint_every=args.checkpoint_every,
        mixed_precision="torch.float16",
    )


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    models = tuple(args.baseline_models or DEFAULT_OMNI_BASELINE_MODELS)
    summary = select_baseline_model(
        runs_root=args.runs_root,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        lang=args.lang,
        model_cards=models,
        max_samples=args.baseline_samples,
    )
    print(summary, flush=True)
    return summary


def run_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    external_root = ensure_external_omnilingual_repo(runs_root=args.runs_root)
    install_colab_omnilingual_dependencies(external_root)
    payload = {
        "external_root": str(external_root),
        "verification": verify_colab_omnilingual_import(external_root),
    }
    save_json(Path(args.runs_root) / "reports" / "omnilingual_bootstrap_summary.json", payload)
    print(payload, flush=True)
    return payload


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    base_model = _selected_model_or_default(Path(args.runs_root))
    prepared = prepare_waxal_omnilingual_dataset(
        runs_root=args.runs_root,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        lang=args.lang,
        cache_dir=args.hf_cache,
        max_duration_seconds=args.max_duration_seconds,
        max_samples_per_split=args.max_samples_per_split,
        base_model=base_model,
    )
    payload = {
        "dataset_root": str(prepared.dataset_root),
        "version_root": str(prepared.version_root),
        "summary_path": str(prepared.summary_path),
        "asset_card_path": str(prepared.asset_card_path),
        "train_config_path": str(prepared.train_config_path),
        "eval_config_path": str(prepared.eval_config_path),
        "split_counts": prepared.split_counts,
        "hours_by_split": prepared.hours_by_split,
    }
    print(payload, flush=True)
    return payload


def run_train(args: argparse.Namespace, *, smoke: bool) -> dict[str, Any]:
    steps = args.smoke_steps if smoke else args.main_steps
    card_path, train_config_path, _ = _prepare_configs(args, steps=steps)
    if args.install_deps:
        external_root = ensure_external_omnilingual_repo(runs_root=args.runs_root)
        install_colab_omnilingual_dependencies(external_root)
        copy_dataset_card_to_external_repo(card_path, external_root)
    if args.skip_train:
        payload = {
            "mode": "train_config_only",
            "train_config_path": str(train_config_path),
            "asset_card_path": str(card_path),
            "steps": steps,
        }
        print(payload, flush=True)
        return payload
    output_dir = run_official_train_recipe(
        runs_root=args.runs_root,
        config_path=train_config_path,
        card_path=card_path,
        install=False,
        output_name="omnilingual_ctc_session_smoke" if smoke else None,
    )
    payload = {"mode": "smoke_train" if smoke else "train", "output_dir": str(output_dir), "steps": steps}
    save_json(Path(args.runs_root) / "reports" / f"{payload['mode']}_summary.json", payload)
    print(payload, flush=True)
    return payload


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    if args.eval_dir:
        payload = collect_ref_hyp_metrics(
            args.eval_dir,
            Path(args.runs_root) / "reports" / "omnilingual_validation_eval.json",
        )
        print(payload, flush=True)
        return payload

    _, _, eval_config_path = _prepare_configs(args, steps=args.main_steps)
    external_root = ensure_external_omnilingual_repo(runs_root=args.runs_root)
    card_path = Path(args.runs_root) / "artifacts" / "omnilingual_generated" / "cards" / "datasets" / f"{DATASET_CARD_NAME}.yaml"
    copy_dataset_card_to_external_repo(card_path, external_root)
    eval_output_dir = Path(args.runs_root) / "runs" / "omnilingual_eval_latest"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "-m",
            "workflows.recipes.wav2vec2.asr.eval",
            str(eval_output_dir),
            "--config-file",
            str(eval_config_path),
        ],
        cwd=external_root,
        env=env_with_external_omnilingual_paths(external_root),
    )
    payload = collect_ref_hyp_metrics(eval_output_dir, Path(args.runs_root) / "reports" / "omnilingual_validation_eval.json")
    print(payload, flush=True)
    return payload


def run_promote(args: argparse.Namespace) -> dict[str, Any]:
    runs_root = Path(args.runs_root)
    promotion_path = runs_root / "reports" / "omnilingual_promotion_summary.json"
    eval_path = Path(args.candidate_eval or runs_root / "reports" / "omnilingual_validation_eval.json")
    candidate_checkpoint = str(args.candidate_checkpoint or runs_root / "runs" / "omnilingual_eval_latest").strip()
    candidate_payload = read_json(eval_path)
    existing = read_json(promotion_path) if promotion_path.exists() else {}
    promote, reason = should_promote_omnilingual(candidate_payload["metrics"], existing.get("best_metrics"))
    decision = {
        "candidate_checkpoint": candidate_checkpoint,
        "candidate_eval": str(eval_path),
        "candidate_metrics": candidate_payload["metrics"],
        "current_best_checkpoint": existing.get("best_checkpoint"),
        "current_best_metrics": existing.get("best_metrics"),
        "promote": bool(promote),
        "reason": reason,
    }
    save_json(runs_root / "reports" / "omnilingual_promotion_decision.json", decision)
    if promote:
        save_json(
            promotion_path,
            {
                "best_checkpoint": candidate_checkpoint,
                "best_metrics": candidate_payload["metrics"],
                "source": "trained_candidate",
                "eval_output_path": str(eval_path),
                "lang": args.lang,
                "reason": reason,
            },
        )
    print(decision, flush=True)
    return decision


def main() -> None:
    args = parse_args()
    _configure_env(args)
    _maybe_install_dependencies(args)
    if args.mode == "bootstrap":
        run_bootstrap(args)
    elif args.mode == "baseline":
        run_baseline(args)
    elif args.mode == "prepare":
        run_prepare(args)
    elif args.mode == "train":
        run_train(args, smoke=False)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "promote":
        run_promote(args)
    elif args.mode == "all":
        run_baseline(args)
        run_prepare(args)
        run_train(args, smoke=True)
        train_payload = run_train(args, smoke=False)
        output_dir = train_payload.get("output_dir")
        final_payload = {"mode": "all_complete", "runs_root": args.runs_root, "train": train_payload}
        if output_dir and (Path(output_dir) / "transcriptions").exists():
            eval_payload = collect_ref_hyp_metrics(
                output_dir,
                Path(args.runs_root) / "reports" / "omnilingual_validation_eval.json",
            )
            args.candidate_checkpoint = output_dir
            args.candidate_eval = str(Path(args.runs_root) / "reports" / "omnilingual_validation_eval.json")
            promotion_payload = run_promote(args)
            final_payload["eval"] = eval_payload
            final_payload["promotion"] = promotion_payload
        else:
            final_payload["next"] = "Run eval with --eval-dir after a recipe output contains transcriptions, then run promote."
        print(final_payload, flush=True)


if __name__ == "__main__":
    main()
