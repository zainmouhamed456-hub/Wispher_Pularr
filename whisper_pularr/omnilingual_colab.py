from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from .eval_utils import compute_error_metrics
from .text import normalize_transcript


DATASET_CARD_NAME = "waxal_ful_pularr"
DEFAULT_OMNI_LANG = "ful_Latn"
DEFAULT_OMNI_BASELINE_MODELS = ("omniASR_CTC_300M", "omniASR_CTC_300M_v2")
DEFAULT_OMNI_MAX_DURATION_SECONDS = 40.0
DEFAULT_OMNI_BASELINE_SAMPLES = 64
DEFAULT_OMNI_MAIN_STEPS = 2000
DEFAULT_OMNI_SMOKE_STEPS = 100
DEFAULT_OMNI_EXTERNAL_REPO = "https://github.com/facebookresearch/omnilingual-asr.git"
DEFAULT_OMNI_EXTERNAL_REF = "main"


@dataclass(frozen=True)
class OmnilingualPreparedDataset:
    dataset_root: Path
    version_root: Path
    summary_path: Path
    asset_card_path: Path
    train_config_path: Path
    eval_config_path: Path
    split_counts: dict[str, int]
    hours_by_split: dict[str, float]


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metrics_sort_key(metrics: dict[str, Any] | None) -> tuple[float, float]:
    if not metrics:
        return (float("inf"), float("inf"))
    return (
        float(metrics.get("normalized_wer", float("inf"))),
        float(metrics.get("normalized_cer", float("inf"))),
    )


def should_promote_omnilingual(
    candidate_metrics: dict[str, Any],
    current_metrics: dict[str, Any] | None,
) -> tuple[bool, str]:
    if not current_metrics:
        return True, "no current promoted checkpoint"

    candidate_wer, candidate_cer = metrics_sort_key(candidate_metrics)
    current_wer, current_cer = metrics_sort_key(current_metrics)
    if candidate_wer < current_wer:
        return True, "normalized WER improved"
    if candidate_wer == current_wer and candidate_cer < current_cer:
        return True, "normalized WER tied and normalized CER improved"
    return False, "candidate did not improve normalized WER or CER tie-break"


def _require_numpy_soundfile() -> tuple[Any, Any]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Dataset conversion requires numpy and soundfile.") from exc
    return np, sf


def _audio_duration_seconds(audio: dict[str, Any]) -> float:
    array = audio.get("array")
    sample_rate = int(audio.get("sampling_rate") or audio.get("sample_rate") or 16000)
    if array is not None:
        return float(len(array) / max(sample_rate, 1))
    duration = audio.get("duration")
    if duration is not None:
        return float(duration)
    return 0.0


def _bytes_to_int8_list(payload: bytes) -> list[int]:
    return [byte if byte < 128 else byte - 256 for byte in payload]


def waxal_row_to_omnilingual_record(
    row: dict[str, Any],
    *,
    split: str,
    lang: str = DEFAULT_OMNI_LANG,
    corpus: str = "waxal",
    max_duration_seconds: float | None = DEFAULT_OMNI_MAX_DURATION_SECONDS,
) -> dict[str, Any] | None:
    text = normalize_transcript(str(row.get("transcription") or row.get("text") or row.get("sentence") or ""))
    if not text:
        return None

    audio = row.get("audio") or {}
    duration_seconds = _audio_duration_seconds(audio)
    if max_duration_seconds is not None and (
        not math.isfinite(duration_seconds) or duration_seconds > float(max_duration_seconds)
    ):
        return None

    array = audio.get("array")
    if array is None:
        return None

    np, sf = _require_numpy_soundfile()
    sample_rate = int(audio.get("sampling_rate") or audio.get("sample_rate") or 16000)
    waveform = np.asarray(array, dtype="float32")
    with BytesIO() as buffer:
        sf.write(buffer, waveform, sample_rate, format="FLAC")
        audio_bytes = buffer.getvalue()

    return {
        "text": text,
        "audio_bytes": _bytes_to_int8_list(audio_bytes),
        "audio_size": int(len(waveform)),
        "corpus": corpus,
        "split": split,
        "language": lang,
    }


def write_omnilingual_parquet_split(
    rows: Iterable[dict[str, Any]],
    *,
    version_root: str | Path,
    split: str,
    lang: str = DEFAULT_OMNI_LANG,
    corpus: str = "waxal",
    part_name: str = "part-00000.parquet",
) -> tuple[Path, int, int]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet writing requires pyarrow.") from exc

    version_root = Path(version_root)
    target_dir = version_root / f"corpus={corpus}" / f"split={split}" / f"language={lang}"
    target_dir.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    schema = pa.schema(
        [
            ("text", pa.string()),
            ("audio_bytes", pa.list_(pa.int8())),
            ("audio_size", pa.int64()),
            ("corpus", pa.dictionary(pa.int32(), pa.string())),
            ("split", pa.dictionary(pa.int32(), pa.string())),
            ("language", pa.dictionary(pa.int32(), pa.string())),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    output_path = target_dir / part_name
    pq.write_table(table, output_path, row_group_size=100)
    audio_size = sum(int(record["audio_size"]) for record in records)
    return output_path, len(records), audio_size


def write_omnilingual_dataset_assets(
    *,
    runs_root: str | Path,
    dataset_name: str = DATASET_CARD_NAME,
    lang: str = DEFAULT_OMNI_LANG,
    base_model: str = "omniASR_CTC_300M",
    max_audio_len: int = 640_000,
    max_num_elements: int = 640_000,
    grad_accumulation: int = 16,
    learning_rate: float = 1e-5,
    num_steps: int = DEFAULT_OMNI_MAIN_STEPS,
    validate_every: int = 250,
    checkpoint_every: int = 500,
    mixed_precision: str = "torch.float16",
) -> tuple[Path, Path, Path]:
    runs_root = Path(runs_root)
    dataset_root = runs_root / "artifacts" / dataset_name
    version_root = dataset_root / "version=0"
    summary_path = dataset_root / "language_distribution_0.tsv"
    generated_root = runs_root / "artifacts" / "omnilingual_generated"
    card_path = generated_root / "cards" / "datasets" / f"{dataset_name}.yaml"
    config_root = generated_root / "configs"
    train_config_path = config_root / "ctc_finetune_pularr.yaml"
    eval_config_path = config_root / "ctc_eval_pularr.yaml"

    tokenizer_name = tokenizer_for_model(base_model)
    for path in (card_path, train_config_path, eval_config_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    card_path.write_text(
        "\n".join(
            [
                f"name: {dataset_name}",
                "dataset_family: mixture_parquet_asr_dataset",
                "dataset_config:",
                f"  data: {version_root.as_posix()}",
                f"tokenizer_ref: {tokenizer_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    common_dataset_yaml = f"""dataset:
  name: "{dataset_name}"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "{summary_path.as_posix()}"
    beta_corpus: 0.5
    beta_language: 0.5
    fragment_streaming:
      partition_filters: 'pc.is_in(pc.field("language"), pa.array(["{lang}"]))'
    fragment_loading:
      cache: True
      num_parallel_fragments: 1
      nb_prefetch: 1
  asr_task_config:
     min_audio_len: 16_000
     max_audio_len: {int(max_audio_len)}
     max_num_elements: {int(max_num_elements)}
     batch_shuffle_window: 1
     normalize_audio: true
     example_shuffle_window: 1
"""
    train_config_path.write_text(
        f"""model:
  name: "{base_model}"

{common_dataset_yaml}
tokenizer:
  name: "{tokenizer_name}"

optimizer:
  config:
    lr: {learning_rate}

trainer:
  freeze_encoder_for_n_steps: 0
  mixed_precision:
    dtype: "{mixed_precision}"
  grad_accumulation:
    num_batches: {int(grad_accumulation)}

regime:
  num_steps: {int(num_steps)}
  validate_after_n_steps: {int(validate_every)}
  validate_every_n_steps: {int(validate_every)}
  checkpoint_after_n_steps: {int(checkpoint_every)}
  checkpoint_every_n_steps: {int(checkpoint_every)}
  publish_metrics_after_n_steps: {int(validate_every)}
  publish_metrics_every_n_steps: {int(validate_every)}
""",
        encoding="utf-8",
    )

    eval_config_path.write_text(
        f"""model:
  name: "{base_model}"

tokenizer:
  name: "{tokenizer_name}"

dataset:
  name: "{dataset_name}"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "{summary_path.as_posix()}"
    beta_corpus: 0.5
    beta_language: 0.5
    fragment_streaming:
      partition_filters: 'pc.is_in(pc.field("language"), pa.array(["{lang}"]))'
    fragment_loading:
      cache: True
      num_parallel_fragments: 1
      nb_prefetch: 1
  asr_task_config:
     min_audio_len: 16_000
     max_audio_len: {int(max_audio_len)}
     max_num_elements: {int(max_num_elements)}
     batch_shuffle_window: 1
     normalize_audio: true
     example_shuffle_window: 1
""",
        encoding="utf-8",
    )
    return card_path, train_config_path, eval_config_path


def tokenizer_for_model(model_card: str) -> str:
    return "omniASR_tokenizer_written_v2" if str(model_card).endswith("_v2") else "omniASR_tokenizer_v1"


def write_language_distribution(
    *,
    dataset_root: str | Path,
    corpus: str,
    lang: str,
    total_audio_size: int,
) -> Path:
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    summary_path = dataset_root / "language_distribution_0.tsv"
    hours = float(total_audio_size) / 16000.0 / 3600.0
    summary_path.write_text(f"corpus\tlanguage\thours\n{corpus}\t{lang}\t{hours:.8f}\n", encoding="utf-8")
    return summary_path


def prepare_waxal_omnilingual_dataset(
    *,
    runs_root: str | Path,
    dataset_name: str,
    dataset_config: str,
    lang: str = DEFAULT_OMNI_LANG,
    cache_dir: str | None = None,
    max_duration_seconds: float = DEFAULT_OMNI_MAX_DURATION_SECONDS,
    max_samples_per_split: int | None = None,
    base_model: str = "omniASR_CTC_300M",
) -> OmnilingualPreparedDataset:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Dataset preparation requires datasets[audio].") from exc

    from .settings import DEFAULT_AUDIO_SAMPLING_RATE

    runs_root = Path(runs_root)
    dataset_root = runs_root / "artifacts" / DATASET_CARD_NAME
    version_root = dataset_root / "version=0"
    split_map = {"train": "train", "validation": "dev", "test": "test"}
    split_counts: dict[str, int] = {}
    hours_by_split: dict[str, float] = {}
    total_audio_size = 0

    for hf_split, omni_split in split_map.items():
        loaded = load_dataset(dataset_name, dataset_config, split=hf_split, cache_dir=cache_dir)
        loaded = loaded.cast_column("audio", __import__("datasets").Audio(sampling_rate=DEFAULT_AUDIO_SAMPLING_RATE))
        if max_samples_per_split is not None:
            loaded = loaded.select(range(min(int(max_samples_per_split), len(loaded))))
        records = []
        for row in loaded:
            record = waxal_row_to_omnilingual_record(
                dict(row),
                split=omni_split,
                lang=lang,
                max_duration_seconds=max_duration_seconds,
            )
            if record is not None:
                records.append(record)
        _, count, audio_size = write_omnilingual_parquet_split(
            records,
            version_root=version_root,
            split=omni_split,
            lang=lang,
        )
        split_counts[omni_split] = count
        hours_by_split[omni_split] = float(audio_size) / 16000.0 / 3600.0
        total_audio_size += audio_size

    summary_path = write_language_distribution(
        dataset_root=dataset_root,
        corpus="waxal",
        lang=lang,
        total_audio_size=total_audio_size,
    )
    card_path, train_config_path, eval_config_path = write_omnilingual_dataset_assets(
        runs_root=runs_root,
        lang=lang,
        base_model=base_model,
        max_audio_len=int(max_duration_seconds * 16000),
    )
    metadata = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "lang": lang,
        "dataset_root": str(dataset_root),
        "version_root": str(version_root),
        "summary_path": str(summary_path),
        "asset_card_path": str(card_path),
        "train_config_path": str(train_config_path),
        "eval_config_path": str(eval_config_path),
        "split_counts": split_counts,
        "hours_by_split": hours_by_split,
    }
    save_json(dataset_root / "preparation_summary.json", metadata)
    return OmnilingualPreparedDataset(
        dataset_root=dataset_root,
        version_root=version_root,
        summary_path=summary_path,
        asset_card_path=card_path,
        train_config_path=train_config_path,
        eval_config_path=eval_config_path,
        split_counts=split_counts,
        hours_by_split=hours_by_split,
    )


def run_command(command: list[str], *, cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def external_omnilingual_python_paths(external_root: str | Path) -> list[Path]:
    external_root = Path(external_root)
    return [external_root / "src", external_root]


def add_external_omnilingual_paths(external_root: str | Path) -> None:
    for path in reversed(external_omnilingual_python_paths(external_root)):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def env_with_external_omnilingual_paths(external_root: str | Path) -> dict[str, str]:
    paths = [str(path) for path in external_omnilingual_python_paths(external_root) if path.exists()]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def ensure_external_omnilingual_repo(
    *,
    runs_root: str | Path,
    repo_url: str = DEFAULT_OMNI_EXTERNAL_REPO,
    ref: str = DEFAULT_OMNI_EXTERNAL_REF,
) -> Path:
    external_root = Path(runs_root) / "external" / "omnilingual-asr"
    if external_root.exists() and (external_root / ".git").exists():
        run_command(["git", "fetch", "--depth", "1", "origin", ref], cwd=external_root)
        run_command(["git", "checkout", "FETCH_HEAD"], cwd=external_root)
        return external_root
    external_root.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", "--depth", "1", "--branch", ref, repo_url, str(external_root)])
    return external_root


def install_colab_omnilingual_dependencies(external_root: str | Path) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
            "torch==2.8.0",
            "torchaudio==2.8.0",
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--extra-index-url",
            "https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu126",
            "-e",
            f"{Path(external_root).as_posix()}[data]",
            "jiwer==3.0.5",
        ]
    )


def verify_colab_omnilingual_import(external_root: str | Path | None = None) -> dict[str, Any]:
    if external_root is not None:
        add_external_omnilingual_paths(external_root)
    try:
        import torch
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    except ImportError as exc:
        raise RuntimeError(
            "Omnilingual import verification failed. Run the bootstrap step first in this Colab runtime."
        ) from exc

    payload = {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "pipeline": ASRInferencePipeline.__name__,
        "torch_version": getattr(torch, "__version__", None),
    }
    return payload


def copy_dataset_card_to_external_repo(card_path: str | Path, external_root: str | Path) -> Path:
    card_path = Path(card_path)
    target = Path(external_root) / "src" / "omnilingual_asr" / "cards" / "datasets" / card_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(card_path.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def evaluate_with_pipeline(
    *,
    model_card: str,
    dataset_name: str,
    dataset_config: str,
    split: str,
    lang: str,
    max_samples: int,
    output_path: str | Path,
    batch_size: int = 1,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset, Audio
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    except ImportError as exc:
        raise RuntimeError("Omnilingual baseline evaluation requires omnilingual-asr and datasets[audio].") from exc

    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    rows = []
    for row in dataset:
        rows.append(dict(row))
        if len(rows) >= max(int(max_samples), 1):
            break
    materialized = __import__("datasets").Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))
    pipeline = ASRInferencePipeline(model_card=model_card, device=None)
    references: list[str] = []
    predictions: list[str] = []
    samples: list[dict[str, Any]] = []
    for start in range(0, len(materialized), max(int(batch_size), 1)):
        batch = list(materialized)[start : start + max(int(batch_size), 1)]
        inputs = [
            {
                "waveform": row["audio"]["array"],
                "sample_rate": int(row["audio"].get("sampling_rate") or 16000),
            }
            for row in batch
        ]
        batch_lang = [lang] * len(inputs) if lang else None
        batch_predictions = pipeline.transcribe(inputs, lang=batch_lang, batch_size=max(int(batch_size), 1))
        for row, prediction_value in zip(batch, batch_predictions):
            reference = str(row.get("transcription") or "").strip()
            prediction = str(prediction_value or "").strip()
            references.append(reference)
            predictions.append(prediction)
            samples.append(
                {
                    "id": row.get("id"),
                    "reference": reference,
                    "prediction": prediction,
                    "normalized_reference": normalize_transcript(reference),
                    "normalized_prediction": normalize_transcript(prediction),
                }
            )
    payload = {"sample_count": len(samples), "metrics": compute_error_metrics(references, predictions), "samples": samples}
    save_json(output_path, payload)
    return payload


def select_baseline_model(
    *,
    runs_root: str | Path,
    dataset_name: str,
    dataset_config: str,
    lang: str,
    model_cards: Iterable[str] = DEFAULT_OMNI_BASELINE_MODELS,
    max_samples: int = DEFAULT_OMNI_BASELINE_SAMPLES,
) -> dict[str, Any]:
    reports_root = Path(runs_root) / "reports"
    results = []
    for model_card in model_cards:
        output_path = reports_root / f"omnilingual_baseline_{model_card}.json"
        try:
            payload = evaluate_with_pipeline(
                model_card=model_card,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split="validation",
                lang=lang,
                max_samples=max_samples,
                output_path=output_path,
            )
            results.append(
                {
                    "model_card": model_card,
                    "output_path": str(output_path),
                    "metrics": payload["metrics"],
                    "sample_count": payload["sample_count"],
                    "ok": True,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model_card": model_card,
                    "output_path": str(output_path),
                    "metrics": None,
                    "sample_count": 0,
                    "ok": False,
                    "error": str(exc),
                }
            )
    successful = [entry for entry in results if entry.get("ok")]
    if not successful:
        save_json(
            reports_root / "omnilingual_baseline_summary.json",
            {"selected_model_card": None, "max_samples": int(max_samples), "lang": lang, "results": results},
        )
        raise RuntimeError("All Omnilingual baseline candidates failed. See omnilingual_baseline_summary.json.")
    successful.sort(key=lambda entry: metrics_sort_key(entry["metrics"]) + (entry["model_card"],))
    summary = {
        "selected_model_card": successful[0]["model_card"],
        "selected_metrics": successful[0]["metrics"],
        "max_samples": int(max_samples),
        "lang": lang,
        "results": results,
    }
    save_json(reports_root / "omnilingual_baseline_summary.json", summary)
    save_json(
        reports_root / "omnilingual_promotion_summary.json",
        {
            "best_checkpoint": successful[0]["model_card"],
            "best_metrics": successful[0]["metrics"],
            "source": "baseline",
            "eval_output_path": successful[0]["output_path"],
            "lang": lang,
        },
    )
    return summary


def run_official_train_recipe(
    *,
    runs_root: str | Path,
    config_path: str | Path,
    card_path: str | Path,
    install: bool = False,
    output_name: str | None = None,
) -> Path:
    external_root = ensure_external_omnilingual_repo(runs_root=runs_root)
    if install:
        install_colab_omnilingual_dependencies(external_root)
    copy_dataset_card_to_external_repo(card_path, external_root)
    sessions_root = Path(runs_root) / "runs"
    sessions_root.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        existing = [path for path in sessions_root.glob("omnilingual_ctc_session_*") if path.is_dir()]
        output_name = f"omnilingual_ctc_session_{len(existing) + 1:03d}"
    output_dir = sessions_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "-m",
            "workflows.recipes.wav2vec2.asr",
            str(output_dir),
            "--config-file",
            str(config_path),
        ],
        cwd=external_root,
        env=env_with_external_omnilingual_paths(external_root),
    )
    return output_dir


def collect_ref_hyp_metrics(eval_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    eval_dir = Path(eval_dir)
    ref_files = sorted((eval_dir / "transcriptions").glob("*.ref.txt"))
    hyp_files = sorted((eval_dir / "transcriptions").glob("*.hyp.txt"))
    references: list[str] = []
    predictions: list[str] = []
    for ref_file, hyp_file in zip(ref_files, hyp_files):
        references.extend(ref_file.read_text(encoding="utf-8").splitlines())
        predictions.extend(hyp_file.read_text(encoding="utf-8").splitlines())
    payload = {
        "sample_count": min(len(references), len(predictions)),
        "metrics": compute_error_metrics(references, predictions),
        "ref_files": [str(path) for path in ref_files],
        "hyp_files": [str(path) for path in hyp_files],
    }
    save_json(output_path, payload)
    return payload
