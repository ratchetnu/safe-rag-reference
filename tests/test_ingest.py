from __future__ import annotations

import pytest

from saferag.config import Settings
from saferag.embeddings import HashingEmbedder
from saferag.indexing import index_documents
from saferag.ingest import IngestionError, chunk_document, load_corpus, parse_document
from saferag.models import Document, Trust
from saferag.store.memory import InMemoryStore
from saferag.text import approx_tokens

from .conftest import CORPUS, LARKSPUR_EMPLOYEE

VALID = """---
doc_id: demo-policy
tenant_id: demo
title: Demo Policy
version: 1
allowed_roles: employee
trust: authoritative
---
# Demo Policy

## First section

Alpha beta gamma. Delta epsilon.

## Second section

Zeta eta theta.
"""


def test_parses_valid_document() -> None:
    doc = parse_document(VALID)
    assert doc.tenant_id == "demo"
    assert doc.allowed_roles == frozenset({"employee"})
    assert doc.trust is Trust.AUTHORITATIVE


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ("tenant_id: demo\n", ""),  # missing tenant must not default to "everyone"
        ("allowed_roles: employee\n", ""),
        ("tenant_id: demo", "tenant_id: Demo Tenant!"),
        ("allowed_roles: employee", "allowed_roles: Employee Role"),
        ("trust: authoritative", "trust: maybe"),
        ("doc_id: demo-policy", "doc_id: ../etc/passwd"),
    ],
)
def test_rejects_bad_access_metadata(needle: str, replacement: str) -> None:
    with pytest.raises(IngestionError):
        parse_document(VALID.replace(needle, replacement))


def test_rejects_missing_front_matter_and_oversize() -> None:
    with pytest.raises(IngestionError):
        parse_document("# no front matter")
    with pytest.raises(IngestionError):
        parse_document(VALID + "x" * 600_000)


def test_chunks_follow_headings_and_budget() -> None:
    settings = Settings(chunk_max_tokens=40)
    doc = parse_document(VALID.replace("Zeta eta theta.", "Zeta eta theta. " * 30))
    chunks = chunk_document(doc, settings)
    assert {c.section for c in chunks} == {"First section", "Second section"}
    assert all(c.chunk_id.startswith("demo-policy#") for c in chunks)
    # Budget is respected except for a single over-long sentence, which is never split.
    assert all(approx_tokens(c.text) <= 40 + 10 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_ids_are_deterministic(settings: Settings) -> None:
    doc = parse_document(VALID)
    assert [c.chunk_id for c in chunk_document(doc, settings)] == [
        c.chunk_id for c in chunk_document(doc, settings)
    ]


def test_corpus_loads_and_every_chunk_carries_access_metadata(docs: list[Document]) -> None:
    assert {d.tenant_id for d in docs} == {"larkspur", "quillfeather"}
    for doc in docs:
        for chunk in chunk_document(doc, Settings()):
            assert chunk.tenant_id == doc.tenant_id
            assert chunk.allowed_roles == doc.allowed_roles


def test_duplicate_doc_ids_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "a.md").write_text(VALID)
    (tmp_path / "b.md").write_text(VALID)
    with pytest.raises(IngestionError):
        load_corpus(tmp_path)


def test_reindex_replaces_stale_chunks(settings: Settings) -> None:
    store = InMemoryStore()
    emb = HashingEmbedder()
    doc = parse_document(VALID.replace("tenant_id: demo", "tenant_id: larkspur"))
    index_documents([doc], store, emb, settings)
    before = len(store)
    edited = Document(**{**doc.__dict__, "body": "## Only section\n\nReplaced text entirely."})
    index_documents([edited], store, emb, settings)
    assert len(store) == 1 < before
    hits = store.keyword_search(LARKSPUR_EMPLOYEE, "alpha beta gamma", 5)
    assert hits == []


def test_corpus_dir_exists() -> None:
    assert CORPUS.is_dir()
