"""Hybrid retrieval: scoped vector + keyword search, fusion, dedupe, rerank."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from enum import StrEnum

from saferag.config import Settings
from saferag.embeddings import Embedder
from saferag.models import Scope, ScopeViolation, ScoredChunk
from saferag.rerank import Reranker
from saferag.store.base import EmbeddingModelMismatch, Store
from saferag.text import tokens


class RetrievalMode(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredChunk]], *, k: int = 60
) -> list[ScoredChunk]:
    """Combine rankings by position, not by raw score.

    Cosine similarity and BM25 live on different, query-dependent scales, so
    adding them directly is fragile. RRF only needs each list's order:
    ``score(d) = sum(1 / (k + rank_i(d)))``.
    """
    fused: dict[str, ScoredChunk] = {}
    totals: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            cid = item.chunk.chunk_id
            totals[cid] = totals.get(cid, 0.0) + 1.0 / (k + rank)
            prev = fused.get(cid)
            signals = {**(prev.signals if prev else {}), **item.signals}
            fused[cid] = ScoredChunk(item.chunk, 0.0, signals)
    out = [
        replace(sc, score=totals[cid], signals={**sc.signals, "rrf": totals[cid]})
        for cid, sc in fused.items()
    ]
    out.sort(key=lambda sc: (-sc.score, sc.chunk.chunk_id))
    return out


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = tokens(text)
    if len(toks) < n:
        return {tuple(toks)}
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def deduplicate(items: Sequence[ScoredChunk], *, threshold: float) -> list[ScoredChunk]:
    """Drop exact (same content hash) and near duplicates, keeping the higher-ranked one.

    Duplicates are common in real corpora (copied paragraphs, FAQ pages that
    restate a policy) and waste context budget without adding evidence.
    """
    kept: list[ScoredChunk] = []
    kept_shingles: list[set[tuple[str, ...]]] = []
    seen_hashes: set[str] = set()
    for item in items:
        if item.chunk.content_hash in seen_hashes:
            continue
        sh = _shingles(item.chunk.text)
        if any(len(sh & other) / len(sh | other) >= threshold for other in kept_shingles):
            continue
        seen_hashes.add(item.chunk.content_hash)
        kept.append(item)
        kept_shingles.append(sh)
    return kept


def cap_per_document(items: Sequence[ScoredChunk], max_per_doc: int) -> list[ScoredChunk]:
    counts: dict[str, int] = {}
    out: list[ScoredChunk] = []
    for item in items:
        n = counts.get(item.chunk.doc_id, 0)
        if n < max_per_doc:
            out.append(item)
            counts[item.chunk.doc_id] = n + 1
    return out


def assert_in_scope(scope: Scope, items: Sequence[ScoredChunk]) -> None:
    """Defence in depth: re-check scope after the store already filtered."""
    for item in items:
        if not scope.permits(item.chunk.tenant_id, item.chunk.allowed_roles):
            raise ScopeViolation("retrieved chunk outside caller scope")


class HybridRetriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        reranker: Reranker | None,
        settings: Settings,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._s = settings

    def check_index_compatibility(self, scope: Scope) -> None:
        indexed = self._store.indexed_model_ids(scope)
        if indexed and indexed != {self._embedder.model_id}:
            raise EmbeddingModelMismatch(
                f"index built with {sorted(indexed)}, query embedder is {self._embedder.model_id}; "
                "re-index before serving"
            )

    def retrieve(
        self,
        scope: Scope,
        query: str,
        *,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        rerank: bool = True,
    ) -> list[ScoredChunk]:
        lists: list[list[ScoredChunk]] = []
        if mode in (RetrievalMode.VECTOR, RetrievalMode.HYBRID):
            self.check_index_compatibility(scope)
            [qvec] = self._embedder.embed([query])
            lists.append(self._store.vector_search(scope, qvec, self._s.vector_k))
        if mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID):
            lists.append(self._store.keyword_search(scope, query, self._s.keyword_k))
        for ranked in lists:
            assert_in_scope(scope, ranked)

        fused = reciprocal_rank_fusion(lists, k=self._s.rrf_k)
        candidates = deduplicate(fused, threshold=self._s.near_duplicate_jaccard)
        candidates = candidates[: self._s.fused_k]
        if rerank and self._reranker is not None:
            candidates = self._reranker.rerank(query, candidates)
        candidates = cap_per_document(candidates, self._s.max_chunks_per_doc)
        return candidates[: self._s.final_k]
