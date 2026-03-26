from __future__ import annotations

import re
from typing import Any

from .text import normalize_transcript

_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)?", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_transcript(text))


def build_label_profile(split: Any) -> dict[str, Any]:
    token_vocab: set[str] = set()
    allowed_chars: set[str] = set()
    for text in split["transcription"]:
        normalized = normalize_transcript(text)
        token_vocab.update(tokenize(normalized))
        allowed_chars.update(char for char in normalized if char.isalpha())
    return {
        "token_vocab": token_vocab,
        "allowed_chars": allowed_chars,
    }


def labeled_token_ratio(text: str, token_vocab: set[str]) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    matched = sum(1 for token in tokens if token in token_vocab)
    return float(matched / len(tokens))


def labeled_char_ratio(text: str, allowed_chars: set[str]) -> float:
    normalized = normalize_transcript(text)
    alpha_chars = [char for char in normalized if char.isalpha()]
    if not alpha_chars:
        return 0.0
    matched = sum(1 for char in alpha_chars if char in allowed_chars)
    return float(matched / len(alpha_chars))


def evaluate_pseudo_label_record(
    record: dict[str, Any],
    *,
    token_vocab: set[str],
    allowed_chars: set[str],
    min_chars: int,
    avg_logprob_threshold: float,
    compression_ratio_threshold: float,
    no_speech_prob_threshold: float,
    min_labeled_token_ratio: float,
    min_labeled_char_ratio: float,
) -> tuple[bool, set[str]]:
    text = str(record.get("pseudo_transcription") or "")
    token_count = len(tokenize(text))
    record["labeled_token_ratio"] = labeled_token_ratio(text, token_vocab)
    record["labeled_char_ratio"] = labeled_char_ratio(text, allowed_chars)

    rejection_reasons: set[str] = set()
    if len(text) < int(min_chars):
        rejection_reasons.add("too_short")
    if float(record.get("avg_logprob", -10.0)) < float(avg_logprob_threshold):
        rejection_reasons.add("low_logprob")
    if float(record.get("compression_ratio", 0.0)) > float(compression_ratio_threshold):
        rejection_reasons.add("high_compression")
    if float(record.get("no_speech_prob", 1.0)) > float(no_speech_prob_threshold):
        rejection_reasons.add("high_no_speech")
    if float(record["labeled_char_ratio"]) < float(min_labeled_char_ratio):
        rejection_reasons.add("language_mismatch")
    if token_count >= 2 and float(record["labeled_token_ratio"]) < float(min_labeled_token_ratio):
        rejection_reasons.add("language_mismatch")
    return (not rejection_reasons, rejection_reasons)
