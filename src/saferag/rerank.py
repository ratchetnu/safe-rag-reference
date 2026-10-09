"""Second-stage reranking.

First-stage retrieval (vector + keyword) is tuned for recall: cheap, wide, and
approximate. A reranker looks at each (query, candidate) pair more carefully and
reorders a short list. In production this is usually a cross-encoder model or a
hosted rerank endpoint. ``LexicalReranker`` is a transparent, deterministic
stand-in that scores term coverage, phrase overlap and heading match.

Reranker scores are also used as the relevance signal for the "no good
evidence -> refuse" decision, because they are comparable across queries in a
way raw RRF or cosine scores are not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import pairwise
from typing import Protocol

from saferag.embeddings import concept_features, query_coverage
from saferag.models import ScoredChunk
from saferag.text import STOPWORDS, content_terms, tokens


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]: ...


def _terms_with_concepts(text: str) -> set[str]:
    words = [w for w in tokens(text) if w not in STOPWORDS]
    return set(content_terms(text)) | set(concept_features(words))


def _bigrams(text: str) -> set[tuple[str, str]]:
    terms = content_terms(text)
    return set(pairwise(terms))


class LexicalReranker:
    def __init__(
        self,
        *,
        w_coverage: float = 0.55,
        w_phrase: float = 0.15,
        w_heading: float = 0.15,
        w_prior: float = 0.15,
    ) -> None:
        self._w = (w_coverage, w_phrase, w_heading, w_prior)

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        q_terms = _terms_with_concepts(query)
        q_bigrams = _bigrams(query)
        max_prior = max(c.score for c in candidates) or 1.0
        w_cov, w_phr, w_head, w_prior = self._w

        rescored: list[ScoredChunk] = []
        for cand in candidates:
            chunk = cand.chunk
            head_terms = _terms_with_concepts(f"{chunk.title} {chunk.section}")
            coverage = query_coverage(query, f"{chunk.section} {chunk.text}")
            phrase = len(q_bigrams & _bigrams(chunk.text)) / len(q_bigrams) if q_bigrams else 0.0
            heading = min(1.0, 2 * len(q_terms & head_terms) / len(q_terms)) if q_terms else 0.0
            prior = cand.score / max_prior
            score = w_cov * coverage + w_phr * phrase + w_head * heading + w_prior * prior
            signals = {**cand.signals, "rerank": score, "coverage": coverage}
            rescored.append(replace(cand, score=score, signals=signals))

        rescored.sort(key=lambda c: (-c.score, c.chunk.chunk_id))
        return rescored
