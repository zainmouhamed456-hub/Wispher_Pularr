from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end Whisper Pularr remote pipeline.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--identity", default=None)
    parser.add_argument("--insecure-host-key-bypass", action="store_true")
    parser.add_argument("--dataset-name", default="google/WaxalNLP")
    parser.add_argument("--dataset-config", default="ful_asr")
    parser.add_argument("--teacher-model", default="openai/whisper-large-v3")
    parser.add_argument("--model-id", default="openai/whisper-small")
    parser.add_argument("--whisper-language", default=None)
    parser.add_argument("--local-report-path", default="reports/hardware_audit.json")
    parser.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu124")
    return parser.parse_args()


def _ssh_base(args: argparse.Namespace) -> list[str]:
    base = ["ssh"]
    if args.identity:
        base.extend(["-i", args.identity])
    if args.insecure_host_key_bypass:
        base.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"])
    base.append(args.host)
    return base


def _scp_base(args: argparse.Namespace) -> list[str]:
    base = ["scp"]
    if args.identity:
        base.extend(["-i", args.identity])
    if args.insecure_host_key_bypass:
        base.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"])
    return base


def run_local(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"Command failed: {command}")
    return completed


def run_remote(args: argparse.Namespace, command: str) -> subprocess.CompletedProcess[str]:
    ssh_command = _ssh_base(args) + [f"bash -lc {shlex.quote(command)}"]
    return run_local(ssh_command)


def resolve_remote_root(args: argparse.Namespace) -> str:
    command = (
        "python3 - <<'PY'\n"
        "import os\n"
        f"print(os.path.abspath(os.path.expanduser({args.remote_root!r})))\n"
        "PY"
    )
    return run_remote(args, command).stdout.strip()


def _should_skip_sync_path(relative: Path) -> bool:
    excluded_parts = {".git", "__pycache__", ".venv", "runs", "artifacts", "reports", "downloads", ".pytest_cache", ".mypy_cache"}
    if any(part in excluded_parts for part in relative.parts):
        return True
    if relative.suffix in {".tar", ".gz", ".zip"}:
        return True
    return False


def sync_repo(args: argparse.Namespace, repo_root: Path, remote_root: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as handle:
        archive_path = Path(handle.name)
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            for path in repo_root.rglob("*"):
                relative = path.relative_to(repo_root)
                if _should_skip_sync_path(relative):
                    continue
                tar.add(path, arcname=str(relative))
        remote_archive = f"{remote_root.rstrip('/')}/project.tar.gz"
        run_remote(args, f"mkdir -p {shlex.quote(remote_root)}")
        run_local(_scp_base(args) + [str(archive_path), f"{args.host}:{remote_archive}"])
        run_remote(args, f"mkdir -p {shlex.quote(remote_root)} && tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_root)}")
    finally:
        archive_path.unlink(missing_ok=True)


def read_remote_json(args: argparse.Namespace, remote_path: str) -> dict:
    completed = run_remote(args, f"cat {shlex.quote(remote_path)}")
    return json.loads(completed.stdout)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    report_path = Path(args.local_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    remote_root = resolve_remote_root(args)

    run_local(
        [
            sys.executable,
            str(repo_root / "hardware_audit.py"),
            "--host",
            args.host,
            "--output",
            str(report_path),
            *([] if not args.identity else ["--identity", args.identity]),
            *(["--insecure-host-key-bypass"] if args.insecure_host_key_bypass else []),
        ],
        cwd=repo_root,
    )

    sync_repo(args, repo_root, remote_root)
    remote_report = f"{remote_root.rstrip('/')}/reports/hardware_audit.json"
    run_remote(args, f"mkdir -p {shlex.quote(remote_root)}/reports")
    run_local(_scp_base(args) + [str(report_path), f"{args.host}:{remote_report}"])
    run_remote(
        args,
        f"bash {shlex.quote(remote_root)}/remote/bootstrap_remote.sh {shlex.quote(remote_root)} {shlex.quote(remote_root + '/.venv')} {shlex.quote(args.torch_index_url)}",
    )

    activate = f"source {shlex.quote(remote_root)}/.venv/bin/activate"
    common = (
        f"{activate} && export HF_HOME={shlex.quote(remote_root)}/hf-cache "
        f"&& export TRANSFORMERS_CACHE={shlex.quote(remote_root)}/hf-cache/transformers "
        f"&& cd {shlex.quote(remote_root)}"
    )

    run_remote(
        args,
        f"{common} && python evaluate.py --checkpoint {shlex.quote(args.model_id)} "
        f"--dataset-name {shlex.quote(args.dataset_name)} --dataset-config {shlex.quote(args.dataset_config)} "
        f"{'' if not args.whisper_language else '--whisper-language ' + shlex.quote(args.whisper_language) + ' '} "
        f"--hardware-report {shlex.quote(remote_report)} --split validation "
        f"--output-path {shlex.quote(remote_root + '/reports/zero_shot_small_validation.json')}",
    )
    run_remote(
        args,
        f"{common} && python evaluate.py --checkpoint {shlex.quote(args.teacher_model)} "
        f"--dataset-name {shlex.quote(args.dataset_name)} --dataset-config {shlex.quote(args.dataset_config)} "
        f"{'' if not args.whisper_language else '--whisper-language ' + shlex.quote(args.whisper_language) + ' '} "
        f"--hardware-report {shlex.quote(remote_report)} --split validation "
        f"--output-path {shlex.quote(remote_root + '/reports/zero_shot_teacher_validation.json')}",
    )

    trials = ["trial_a", "trial_b", "trial_c"]
    summaries: dict[str, dict] = {}
    for trial in trials:
        run_remote(
            args,
            f"{common} && python train.py --stage supervised --preset {trial} "
            f"--dataset-name {shlex.quote(args.dataset_name)} --dataset-config {shlex.quote(args.dataset_config)} "
            f"{'' if not args.whisper_language else '--whisper-language ' + shlex.quote(args.whisper_language) + ' '} "
            f"--model-id {shlex.quote(args.model_id)} --hardware-report {shlex.quote(remote_report)} "
            f"--output-dir {shlex.quote(remote_root + '/runs/' + trial)}",
        )
        summary = read_remote_json(args, f"{remote_root}/runs/{trial}/run_summary.json")
        summaries[trial] = summary

    best_trial = min(
        summaries,
        key=lambda trial: (
            float(summaries[trial]["best_metrics"]["normalized_wer"]),
            float(summaries[trial]["best_metrics"]["normalized_cer"]),
        ),
    )
    best_checkpoint = remote_root + "/runs/" + best_trial + "/best_full_eval"
    run_remote(
        args,
        f"{common} && python pseudo_label.py --dataset-name {shlex.quote(args.dataset_name)} "
        f"--dataset-config {shlex.quote(args.dataset_config)} --teacher-model {shlex.quote(args.teacher_model)} "
        f"{'' if not args.whisper_language else '--whisper-language ' + shlex.quote(args.whisper_language) + ' '} "
        f"--hardware-report {shlex.quote(remote_report)} "
        f"--output-path {shlex.quote(remote_root + '/artifacts/pseudo_labels.jsonl')}",
    )
    run_remote(
        args,
        f"{common} && python run_self_train_sequence.py "
        f"--dataset-name {shlex.quote(args.dataset_name)} --dataset-config {shlex.quote(args.dataset_config)} "
        f"--model-id {shlex.quote(args.model_id)} "
        f"{'' if not args.whisper_language else '--whisper-language ' + shlex.quote(args.whisper_language) + ' '} "
        f"--base-checkpoint {shlex.quote(best_checkpoint)} "
        f"--manifests-dir {shlex.quote(remote_root + '/artifacts/pseudo_labels_manifests')} "
        f"--hardware-report {shlex.quote(remote_report)} "
        f"--output-root {shlex.quote(remote_root + '/runs/self_train_snapshots')}",
    )
    sequence_summary = read_remote_json(args, f"{remote_root}/runs/self_train_snapshots/sequence_summary.json")
    run_remote(
        args,
        f"{common} && python evaluate.py --checkpoint {shlex.quote(sequence_summary['last_best_full_eval_dir'])} "
        f"--dataset-name {shlex.quote(args.dataset_name)} --dataset-config {shlex.quote(args.dataset_config)} "
        f"{'' if not args.whisper_language else '--whisper-language ' + shlex.quote(args.whisper_language) + ' '} "
        f"--hardware-report {shlex.quote(remote_report)} --split test "
        f"--output-path {shlex.quote(remote_root + '/reports/final_test_eval.json')}",
    )

    final_summary = {
        "best_supervised_trial": best_trial,
        "best_supervised_summary": summaries[best_trial],
        "remote_root": remote_root,
        "remote_report": remote_report,
    }
    print(json.dumps(final_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
