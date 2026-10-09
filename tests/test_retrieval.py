from __future__ import annotations

import pytest

from saferag.config import Settings
from saferag.embeddings import HashingEmbedder
from saferag.models import Chunk, ScoredChunk, Trust
from saferag.rerank import LexicalReranker
from saferag.retrieval import (
    HybridRetriever,
    RetrievalMode,
    cap_per_document,
    deduplicate,
    reciprocal_rank_fusion,
)
from saferag.store.base import EmbeddingModelMismatch
from saferag.store.memory import InMemoryStore

from .conftest import LARKSPUR_EMPLOYEE


def _chunk(cid: str, text: str = "x", doc: str = "d") -> Chunk:
    return Chunk(
        cid,
        doc,
        "t1",
        frozenset({"employee"}),
        Trust.AUTHORITATIVE,
        "T",
        "1",
        "S",
        "s",
        0,
        text,
        cid,
    )


def _sc(cid: str, score: float = 1.0, **kw: str) -> ScoredChunk:
    return ScoredChunk(_chunk(cid, **kw), score)


def test_rrf_rewards_agreement_between_lists() -> None:
    vector = [_sc("a"), _sc("b"), _sc("c")]
    keyword = [_sc("c"), _sc("b"), _sc("d")]
    fused = [s.chunk.chunk_id for s in reciprocal_rank_fusion([vector, keyword], k=60)]
    assert set(fused[:2]) == {"b", "c"}  # present in both lists beats a single first place
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_score_formula() -> None:
    fused = reciprocal_rank_fusion([[_sc("a")], [_sc("b"), _sc("a")]], k=60)
    a = next(s for s in fused if s.chunk.chunk_id == "a")
    assert a.score == pytest.approx(1 / 61 + 1 / 62)


def test_deduplicate_drops_exact_and_near_duplicates() -> None:
    base = "the hotel limit is 220 per night in most cities and 300 in high cost cities"
    items = [
        _sc("a", text=base),
        _sc("b", text=base + " today"),
        _sc("c", text="completely different text about passwords and laptops"),
    ]
    kept = [s.chunk.chunk_id for s in deduplicate(items, threshold=0.85)]
    assert kept == ["a", "c"]


def test_cap_per_document() -> None:
    items = [_sc(f"c{i}", doc="same") for i in range(5)] + [_sc("z", doc="other")]
    assert len(cap_per_document(items, 2)) == 3


def test_hybrid_dedupes_copied_faq_paragraph(retriever: HybridRetriever) -> None:
    hits = retriever.retrieve(LARKSPUR_EMPLOYEE, "What is the hotel limit per night?")
    lodging = [h for h in hits if "220" in h.chunk.text]
    assert len(lodging) == 1


def _rank(hits: list[ScoredChunk], slug: str) -> int | None:
    slugs = [h.chunk.section_slug for h in hits]
    return slugs.index(slug) + 1 if slug in slugs else None


def test_keyword_rescues_identifier_lookup(retriever: HybridRetriever) -> None:
    # Dense-style vectors under-weight identifiers; exact keyword match does not.
    q = "VSQ-3"
    vec = retriever.retrieve(LARKSPUR_EMPLOYEE, q, mode=RetrievalMode.VECTOR, rerank=False)
    kw = retriever.retrieve(LARKSPUR_EMPLOYEE, q, mode=RetrievalMode.KEYWORD, rerank=False)
    hy = retriever.retrieve(LARKSPUR_EMPLOYEE, q)
    assert _rank(kw, "security-review") == 1
    assert _rank(vec, "security-review") != 1
    assert _rank(hy, "security-review") == 1


def test_vector_rescues_paraphrase(retriever: HybridRetriever) -> None:
    # No word in the question appears in the relevant section; only meaning connects them.
    q = "How much can I spend on dinner while abroad?"
    vec = retriever.retrieve(LARKSPUR_EMPLOYEE, q, mode=RetrievalMode.VECTOR, rerank=False)
    kw = retriever.retrieve(LARKSPUR_EMPLOYEE, q, mode=RetrievalMode.KEYWORD, rerank=False)
    hy = retriever.retrieve(LARKSPUR_EMPLOYEE, q)
    assert _rank(vec, "meal-limits") == 1
    assert _rank(kw, "meal-limits") is None
    assert _rank(hy, "meal-limits") == 1


def test_embedding_model_change_requires_reindex(store: InMemoryStore, settings: Settings) -> None:
    retriever = HybridRetriever(store, HashingEmbedder(dim=256), LexicalReranker(), settings)
    with pytest.raises(EmbeddingModelMismatch):
        retriever.retrieve(LARKSPUR_EMPLOYEE, "meal limit")
    # Keyword-only search does not depend on vectors and keeps working during a re-index.
    assert retriever.retrieve(LARKSPUR_EMPLOYEE, "meal limit", mode=RetrievalMode.KEYWORD)


def test_results_are_bounded(retriever: HybridRetriever, settings: Settings) -> None:
    hits = retriever.retrieve(LARKSPUR_EMPLOYEE, "policy expense travel vendor password")
    assert len(hits) <= settings.final_k
    per_doc: dict[str, int] = {}
    for h in hits:
        per_doc[h.chunk.doc_id] = per_doc.get(h.chunk.doc_id, 0) + 1
    assert max(per_doc.values()) <= settings.max_chunks_per_doc
