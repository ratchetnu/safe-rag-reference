"""Index building and convenience factories."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from saferag.config import Settings
from saferag.embeddings import Embedder, HashingEmbedder
from saferag.ingest import chunk_document, load_corpus
from saferag.models import Document
from saferag.pipeline import RagPipeline
from saferag.providers.base import LLMProvider
from saferag.providers.extractive import ExtractiveProvider
from saferag.providers.resilience import CircuitBreaker, ResilientProvider
from saferag.rerank import LexicalReranker
from saferag.retrieval import HybridRetriever
from saferag.store.base import Store
from saferag.store.memory import InMemoryStore


@dataclass(frozen=True)
class IndexReport:
    documents: int
    chunks: int
    embedding_model: str


def index_documents(
    docs: Iterable[Document], store: Store, embedder: Embedder, settings: Settings
) -> IndexReport:
    """(Re)index documents. Each document's chunks are replaced atomically, so an
    edited document never leaves stale chunks behind."""
    n_docs = n_chunks = 0
    for doc in docs:
        chunks = chunk_document(doc, settings)
        vectors = embedder.embed([c.embedding_text() for c in chunks])
        n_chunks += store.replace_document_chunks(
            doc.tenant_id, doc.doc_id, chunks, vectors, embedder.model_id
        )
        n_docs += 1
    return IndexReport(n_docs, n_chunks, embedder.model_id)


def default_corpus_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "corpus"


def wrap_provider(
    provider: LLMProvider, settings: Settings, *, sleep: Callable[[float], None] = time.sleep
) -> ResilientProvider:
    return ResilientProvider(
        provider,
        max_retries=settings.provider_max_retries,
        sleep=sleep,
        breaker=CircuitBreaker(
            failure_threshold=settings.breaker_failure_threshold,
            cooldown_s=settings.breaker_cooldown_s,
        ),
    )


def build_pipeline(
    *,
    store: Store,
    provider: LLMProvider | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> RagPipeline:
    settings = settings or Settings()
    embedder = embedder or HashingEmbedder()
    retriever = HybridRetriever(store, embedder, LexicalReranker(), settings)
    wrapped = wrap_provider(provider or ExtractiveProvider(), settings, sleep=sleep)
    return RagPipeline(retriever, wrapped, settings)


def build_offline_pipeline(
    corpus_dir: Path | None = None, settings: Settings | None = None
) -> tuple[RagPipeline, InMemoryStore]:
    settings = settings or Settings()
    store = InMemoryStore()
    index_documents(
        load_corpus(corpus_dir or default_corpus_dir()), store, HashingEmbedder(), settings
    )
    return build_pipeline(store=store, settings=settings), store
