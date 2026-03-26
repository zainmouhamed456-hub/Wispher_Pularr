from __future__ import annotations

from typing import Any


def resolve_whisper_language(tokenizer: Any, language: str | None) -> str | None:
    cleaned = str(language or "").strip()
    if not cleaned:
        return None

    lang_to_id = getattr(tokenizer, "lang_to_id", None)
    if not isinstance(lang_to_id, dict) or not lang_to_id:
        return cleaned

    if cleaned in lang_to_id:
        return cleaned

    lowered_map = {key.lower(): key for key in lang_to_id}
    lowered = cleaned.lower()
    if lowered in lowered_map:
        return lowered_map[lowered]

    bare = cleaned.strip("<>|").lower()
    bare_map = {key.strip("<>|").lower(): key for key in lang_to_id}
    return bare_map.get(bare)


def configure_whisper_prompt(processor: Any, model: Any | None = None, language: str | None = None) -> None:
    tokenizer = processor.tokenizer
    resolved_language = resolve_whisper_language(tokenizer, language)

    if hasattr(tokenizer, "set_prefix_tokens"):
        if resolved_language:
            tokenizer.set_prefix_tokens(task="transcribe", language=resolved_language)
        else:
            tokenizer.set_prefix_tokens(task="transcribe")

    if model is not None and hasattr(model, "generation_config"):
        model.generation_config.task = "transcribe"
        model.generation_config.language = resolved_language
        model.generation_config.return_timestamps = False
        model.generation_config.forced_decoder_ids = None
    if model is not None and hasattr(model, "config") and hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None
