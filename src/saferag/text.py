"""Small, dependency-free text utilities used by chunking, BM25 and the offline embedder.

These are intentionally simple and deterministic. A production system would use
the tokenizer of its embedding model and a proper stemmer or analyzer.
"""

from __future__ import annotations

import math
import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*(?:-[a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# Zero-width and bidi control characters are a common way to hide instructions.
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")

STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could do does did for from had has have how
    i if in into is it its may me might must my of on or our shall should so than that the
    their them then there these they this those to was we were what when where which who
    whom why will with would you your yours about any all also am per via up out over under
    many much get gets got need needs policy policies please tell know
    """.split()
)

_SUFFIXES = (
    "ements",
    "ement",
    "ments",
    "ment",
    "ations",
    "ation",
    "ings",
    "ing",
    "als",
    "al",
    "ies",
    "ed",
    "es",
    "s",
)


def normalize(text: str) -> str:
    """NFKC-normalise, drop invisible control characters, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def stem(word: str) -> str:
    if any(ch.isdigit() for ch in word):
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokens(text: str) -> list[str]:
    """Lowercase word tokens.

    Thousands separators are removed from numbers (``2,000`` -> ``2000``).
    Hyphenated tokens that contain a digit are kept whole because they are
    usually identifiers (``EXP-14``, ``SEV1``); other hyphenated words are split.
    """
    out: list[str] = []
    for raw in _TOKEN_RE.findall(normalize(text).lower()):
        tok = raw.replace(",", "").rstrip(".")
        if "-" in tok and not any(ch.isdigit() for ch in tok):
            out.extend(part for part in tok.split("-") if part)
        elif tok:
            out.append(tok)
    return out


def content_terms(text: str) -> list[str]:
    """Stemmed tokens with stopwords removed. Used for BM25 and overlap scoring."""
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


def sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(normalize(text))
    return [p.strip() for p in parts if p.strip()]


def approx_tokens(text: str) -> int:
    """Rough token estimate (about four characters per token for English).

    Good enough for budget enforcement with a safety margin. Use the provider's
    token counter when exact numbers matter for billing.
    """
    return max(1, math.ceil(len(text) / 4))


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize(text).lower()).strip("-")
    return slug or "section"


def numbers_in(text: str) -> set[str]:
    """Numeric literals, normalised (``$2,000`` -> ``2000``)."""
    return {t for t in tokens(text) if any(ch.isdigit() for ch in t)}
