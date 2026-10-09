from __future__ import annotations

import logging
from pathlib import Path

import pytest

from saferag.config import Settings
from saferag.embeddings import HashingEmbedder
from saferag.indexing import index_documents
from saferag.ingest import load_corpus
from saferag.models import Document, Scope
from saferag.rerank import LexicalReranker
from saferag.retrieval import HybridRetriever
from saferag.store.memory import InMemoryStore

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"

LARKSPUR_EMPLOYEE = Scope("larkspur", frozenset({"employee"}))
LARKSPUR_SECURITY = Scope("larkspur", frozenset({"security"}))
QUILLFEATHER_EMPLOYEE = Scope("quillfeather", frozenset({"employee"}))


@pytest.fixture(autouse=True)
def _quiet_audit_logs() -> None:
    logging.getLogger("saferag.audit").setLevel(logging.CRITICAL + 1)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def docs() -> list[Document]:
    return load_corpus(CORPUS)


@pytest.fixture
def store(docs: list[Document], settings: Settings) -> InMemoryStore:
    s = InMemoryStore()
    index_documents(docs, s, HashingEmbedder(), settings)
    return s


@pytest.fixture
def retriever(store: InMemoryStore, settings: Settings) -> HybridRetriever:
    return HybridRetriever(store, HashingEmbedder(), LexicalReranker(), settings)
