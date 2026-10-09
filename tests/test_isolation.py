"""Tenant and role isolation. These are the properties that must never regress."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from saferag.config import Settings
from saferag.embeddings import HashingEmbedder, Vector
from saferag.indexing import build_pipeline
from saferag.models import AnswerStatus, Chunk, Scope, ScopeError, ScopeViolation, ScoredChunk
from saferag.retrieval import HybridRetriever, RetrievalMode, assert_in_scope
from saferag.store.memory import InMemoryStore

from .conftest import LARKSPUR_EMPLOYEE, LARKSPUR_SECURITY, QUILLFEATHER_EMPLOYEE

PROBES = [
    "meal limit",
    "hotel per night",
    "password length",
    "EXP-14",
    "QL-EXP-2",
    "VSQ-3",
    "QF-VRA",
    "incident bridge code",
    "SEV1 paging",
    "vendor payment terms Net",
]


@pytest.mark.parametrize(
    "bad",
    [
        ("", {"employee"}),
        ("UPPER", {"employee"}),
        ("ok-tenant", set()),
        ("ok-tenant", {"Bad Role"}),
    ],
)
def test_scope_validation(bad: tuple[str, set[str]]) -> None:
    with pytest.raises(ScopeError):
        Scope(bad[0], frozenset(bad[1]))


@pytest.mark.parametrize("scope", [LARKSPUR_EMPLOYEE, QUILLFEATHER_EMPLOYEE, LARKSPUR_SECURITY])
@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_retrieval_never_returns_other_tenants(
    retriever: HybridRetriever, scope: Scope, mode: RetrievalMode
) -> None:
    for q in PROBES:
        for hit in retriever.retrieve(scope, q, mode=mode):
            assert hit.chunk.tenant_id == scope.tenant_id
            assert scope.roles & hit.chunk.allowed_roles


def test_restricted_document_requires_role(retriever: HybridRetriever) -> None:
    q = "What is the incident bridge code?"
    emp = retriever.retrieve(LARKSPUR_EMPLOYEE, q)
    sec = retriever.retrieve(LARKSPUR_SECURITY, q)
    assert all(h.chunk.doc_id != "larkspur-incident-runbook" for h in emp)
    assert sec[0].chunk.doc_id == "larkspur-incident-runbook"


def test_bm25_statistics_are_computed_within_scope(store: InMemoryStore) -> None:
    # Identical question, different tenants -> each gets only its own rows and scores
    # derived only from its own corpus.
    a = store.keyword_search(LARKSPUR_EMPLOYEE, "hotel limit", 3)
    b = store.keyword_search(QUILLFEATHER_EMPLOYEE, "hotel limit", 3)
    assert {h.chunk.tenant_id for h in a} == {"larkspur"}
    assert {h.chunk.tenant_id for h in b} == {"quillfeather"}


def test_assert_in_scope_raises_on_foreign_chunk(store: InMemoryStore) -> None:
    foreign = store.keyword_search(QUILLFEATHER_EMPLOYEE, "hotel", 1)
    with pytest.raises(ScopeViolation):
        assert_in_scope(LARKSPUR_EMPLOYEE, foreign)


class LeakyStore(InMemoryStore):
    """Simulates a storage bug: ignores the scope filter entirely."""

    _everyone = Scope("larkspur", frozenset({"employee"}))

    def vector_search(self, scope: Scope, vector: Vector, k: int) -> list[ScoredChunk]:
        rows = [r for r in self._rows.values()]
        sims = [(float(r.vector @ vector), r.chunk) for r in rows]
        sims.sort(key=lambda t: -t[0])
        return [ScoredChunk(c, s, {"vector": s}) for s, c in sims[:k]]

    def keyword_search(self, scope: Scope, query: str, k: int) -> list[ScoredChunk]:
        return []

    def indexed_model_ids(self, scope: Scope) -> set[str]:
        return {r.model_id for r in self._rows.values()}


def test_pipeline_fails_closed_when_store_leaks(docs: Sequence[object], settings: Settings) -> None:
    from saferag.indexing import index_documents

    leaky = LeakyStore()
    index_documents(docs, leaky, HashingEmbedder(), settings)  # type: ignore[arg-type]
    pipeline = build_pipeline(store=leaky, settings=settings)
    result = pipeline.answer(QUILLFEATHER_EMPLOYEE, "What is the meal limit?")
    assert result.status is AnswerStatus.REFUSED
    assert result.reason == "internal_error"
    assert result.citations == ()
    assert result.trace.retrieved_ids == ()


def test_chunk_type_is_frozen() -> None:
    import dataclasses

    assert dataclasses.is_dataclass(Chunk)
    assert Chunk.__dataclass_params__.frozen  # type: ignore[attr-defined]
