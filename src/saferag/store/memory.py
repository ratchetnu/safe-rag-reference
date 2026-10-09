"""In-memory store with brute-force cosine search and BM25.

Used by tests and the offline evaluation. It implements the same pre-filtering
contract as the Postgres store, and BM25 statistics are computed only over the
caller's visible rows so term statistics from other tenants cannot influence
(or be inferred from) ranking.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from saferag.embeddings import Vector
from saferag.models import Chunk, Scope, ScoredChunk
from saferag.text import content_terms


@dataclass
class _Row:
    chunk: Chunk
    vector: Vector
    model_id: str
    terms: list[str]


class InMemoryStore:
    def __init__(self, *, bm25_k1: float = 1.2, bm25_b: float = 0.75) -> None:
        self._rows: dict[str, _Row] = {}
        self._k1 = bm25_k1
        self._b = bm25_b

    def __len__(self) -> int:
        return len(self._rows)

    def replace_document_chunks(
        self,
        tenant_id: str,
        doc_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        model_id: str,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        if any(c.tenant_id != tenant_id or c.doc_id != doc_id for c in chunks):
            raise ValueError("chunk does not belong to the document being replaced")
        self.delete_document(tenant_id, doc_id)
        for chunk, vec in zip(chunks, vectors, strict=True):
            terms = content_terms(f"{chunk.title} {chunk.section} {chunk.text}")
            self._rows[f"{tenant_id}/{chunk.chunk_id}"] = _Row(chunk, vec, model_id, terms)
        return len(chunks)

    def delete_document(self, tenant_id: str, doc_id: str) -> int:
        doomed = [
            key
            for key, row in self._rows.items()
            if row.chunk.tenant_id == tenant_id and row.chunk.doc_id == doc_id
        ]
        for key in doomed:
            del self._rows[key]
        return len(doomed)

    def _visible(self, scope: Scope) -> list[_Row]:
        return [
            r
            for r in self._rows.values()
            if scope.permits(r.chunk.tenant_id, r.chunk.allowed_roles)
        ]

    def indexed_model_ids(self, scope: Scope) -> set[str]:
        return {r.model_id for r in self._visible(scope)}

    def vector_search(self, scope: Scope, vector: Vector, k: int) -> list[ScoredChunk]:
        rows = self._visible(scope)
        if not rows:
            return []
        matrix = np.stack([r.vector for r in rows])
        sims = matrix @ vector
        order = np.argsort(-sims, kind="stable")[:k]
        return [
            ScoredChunk(rows[i].chunk, float(sims[i]), {"vector": float(sims[i])}) for i in order
        ]

    def keyword_search(self, scope: Scope, query: str, k: int) -> list[ScoredChunk]:
        rows = self._visible(scope)
        q_terms = set(content_terms(query))
        if not rows or not q_terms:
            return []
        n = len(rows)
        avg_len = sum(len(r.terms) for r in rows) / n
        df: Counter[str] = Counter()
        for r in rows:
            df.update(set(r.terms) & q_terms)
        scored: list[tuple[float, int]] = []
        for idx, r in enumerate(rows):
            tf = Counter(t for t in r.terms if t in q_terms)
            score = 0.0
            for term, freq in tf.items():
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                denom = freq + self._k1 * (1 - self._b + self._b * len(r.terms) / avg_len)
                score += idf * freq * (self._k1 + 1) / denom
            if score > 0:
                scored.append((score, idx))
        scored.sort(key=lambda s: (-s[0], rows[s[1]].chunk.chunk_id))
        return [ScoredChunk(rows[i].chunk, score, {"keyword": score}) for score, i in scored[:k]]
