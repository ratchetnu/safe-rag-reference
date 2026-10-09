"""Embedding interface plus a deterministic offline embedder.

``HashingEmbedder`` is a stand-in so that tests and the evaluation harness run
without network access or model downloads. It is NOT a semantic model:

* Words, word bigrams and character trigrams are hashed into a fixed-size vector
  (feature hashing). Character trigrams give some tolerance to inflection.
* Tokens containing digits (amounts, form codes like ``EXP-14``) get a low
  weight. Dense embedding models are known to represent rare identifiers and
  exact numbers poorly; this reproduces that weakness so the offline
  evaluation shows why keyword search still matters.
* A tiny hand-written concept table maps a few synonyms ("vacation" -> PTO) to
  shared features. This imitates, very crudely, the one property of real
  embeddings that matters for the retrieval story: related wording lands near
  each other even with no exact word overlap.

Swap in a real embedding model by implementing the ``Embedder`` protocol. The
``model_id`` is stored next to each vector so that a model change is detected
and forces re-indexing instead of silently mixing incompatible vector spaces.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np
import numpy.typing as npt

from saferag.text import STOPWORDS, stem, tokens

Vector = npt.NDArray[np.float32]

# Synthetic "semantic" layer. Kept small and visible on purpose.
CONCEPTS: dict[str, str] = {
    "vacation": "concept-pto",
    "holiday": "concept-pto",
    "pto": "concept-pto",
    "time-off": "concept-pto",
    "rollover": "concept-carryover",
    "roll": "concept-carryover",
    "carryover": "concept-carryover",
    "carry": "concept-carryover",
    "reimburse": "concept-reimburse",
    "reimbursable": "concept-reimburse",
    "refund": "concept-reimburse",
    "expense": "concept-reimburse",
    "hotel": "concept-lodging",
    "lodging": "concept-lodging",
    "accommodation": "concept-lodging",
    "night": "concept-lodging",
    "flight": "concept-air",
    "fly": "concept-air",
    "airfare": "concept-air",
    "cabin": "concept-air",
    "password": "concept-credential",
    "passphrase": "concept-credential",
    "credential": "concept-credential",
    "vendor": "concept-supplier",
    "supplier": "concept-supplier",
    "phishing": "concept-phish",
    "phish": "concept-phish",
    "scam": "concept-phish",
    "suspicious": "concept-phish",
    "mileage": "concept-driving",
    "car": "concept-driving",
    "vehicle": "concept-driving",
    "drive": "concept-driving",
    "approve": "concept-approval",
    "approval": "concept-approval",
    "authorize": "concept-approval",
    "sign": "concept-approval",
    "meal": "concept-food",
    "food": "concept-food",
    "dinner": "concept-food",
    "eat": "concept-food",
    "lunch": "concept-food",
    "sick": "concept-illness",
    "ill": "concept-illness",
    "illness": "concept-illness",
    "usb": "concept-removable",
    "thumb": "concept-removable",
    "removable": "concept-removable",
    "bid": "concept-quote",
    "quote": "concept-quote",
    "tender": "concept-quote",
    "limit": "concept-limit",
    "cap": "concept-limit",
    "maximum": "concept-limit",
    "max": "concept-limit",
    "allowed": "concept-limit",
    "allowance": "concept-limit",
    "rate": "concept-limit",
    "minimum": "concept-minimum",
    "least": "concept-minimum",
    "length": "concept-length",
    "long": "concept-length",
    "characters": "concept-length",
    "pay": "concept-payment",
    "payment": "concept-payment",
    "invoice": "concept-payment",
}


_STEMMED_CONCEPTS = {stem(k): v for k, v in CONCEPTS.items()}


def concept_of(word: str) -> str | None:
    return CONCEPTS.get(word) or _STEMMED_CONCEPTS.get(stem(word))


def concept_features(words: Sequence[str]) -> list[str]:
    return [c for c in (concept_of(w) for w in words) if c]


def query_coverage(query: str, text: str) -> float:
    """Fraction of the query's content words that ``text`` covers.

    A word counts as covered if its stem appears in ``text`` or if it maps to a
    concept that ``text`` also expresses. Concepts substitute for a word; they
    never add extra credit, so a single shared synonym cannot dominate.
    """
    q_words = list(dict.fromkeys(w for w in tokens(query) if w not in STOPWORDS))
    if not q_words:
        return 0.0
    t_words = [w for w in tokens(text) if w not in STOPWORDS]
    t_stems = {stem(w) for w in t_words}
    t_concepts = set(concept_features(t_words))
    covered = 0
    for word in q_words:
        concept = concept_of(word)
        if stem(word) in t_stems or (concept is not None and concept in t_concepts):
            covered += 1
    return covered / len(q_words)


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def model_id(self) -> str:
        return f"hashing-v1-d{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self._dim, (1.0 if (value >> 63) & 1 else -1.0)

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = [w for w in tokens(text) if w not in STOPWORDS]
        stems = [stem(w) for w in words]
        feats: list[tuple[str, float]] = [
            (f"w:{s}", 0.15 if any(ch.isdigit() for ch in s) else 1.0) for s in stems
        ]
        feats += [(f"b:{a}_{b}", 0.5) for a, b in pairwise(stems)]
        for s in stems:
            if any(ch.isdigit() for ch in s):
                continue
            padded = f"#{s}#"
            feats += [(f"c:{padded[i : i + 3]}", 0.25) for i in range(len(padded) - 2)]
        feats += [(f"k:{c}", 1.5) for c in concept_features(words)]
        return feats

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            for feature, weight in self._features(text):
                idx, sign = self._bucket(feature)
                vec[idx] += sign * weight
            norm = float(np.linalg.norm(vec))
            out.append(vec / norm if norm > 0 else vec)
        return out
