from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_APOSTROPHE_RE = re.compile(r"(?<=\w)\s*'\s*(?=\w)", re.UNICODE)
_HYPHEN_RE = re.compile(r"(?<=\w)\s*-\s*(?=\w)", re.UNICODE)
_DROP_PUNCT_RE = re.compile(r"[^\w\s'-]", re.UNICODE)
_STRIP_EDGE_MARK_RE = re.compile(r"(^|(?<=\s))['-]+|['-]+(?=$|(?=\s))", re.UNICODE)

_CHAR_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "_": " ",
    }
)


def normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_CHAR_REPLACEMENTS)
    text = text.lower().strip()
    text = _APOSTROPHE_RE.sub("'", text)
    text = _HYPHEN_RE.sub("-", text)
    text = _DROP_PUNCT_RE.sub(" ", text)
    text = _STRIP_EDGE_MARK_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def compression_ratio(text: str) -> float:
    normalized = normalize_transcript(text)
    if not normalized:
        return 0.0
    import zlib

    compressed = zlib.compress(normalized.encode("utf-8"))
    return len(normalized.encode("utf-8")) / max(len(compressed), 1)
