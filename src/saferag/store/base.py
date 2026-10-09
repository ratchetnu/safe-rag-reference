"""Storage interface.

Every read method takes a ``Scope`` and MUST apply it as a filter *before*
ranking (pre-filtering). Ranking first and filtering afterwards both wastes the
top-k budget on rows the caller can never see and risks leaking them if the
post-filter is forgotten.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from saferag.embeddings import Vector
from saferag.models import Chunk, Scope, ScoredChunk


class EmbeddingModelMismatch(RuntimeError):
    """The index was built with a different embedding model than the query uses."""


class Store(Protocol):
    def replace_document_chunks(
        self,
        tenant_id: str,
        doc_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        model_id: str,
    ) -> int:
        """Atomically replace all chunks of one document. Returns rows written."""
        ...

    def delete_document(self, tenant_id: str, doc_id: str) -> int: ...

    def vector_search(self, scope: Scope, vector: Vector, k: int) -> list[ScoredChunk]: ...

    def keyword_search(self, scope: Scope, query: str, k: int) -> list[ScoredChunk]: ...

    def indexed_model_ids(self, scope: Scope) -> set[str]: ...
