"""PostgreSQL + pgvector store.

* Vector search: cosine distance over an HNSW index, filtered by tenant and role.
* Keyword search: Postgres full-text search (``tsvector`` + ``ts_rank_cd``).
* Isolation: explicit WHERE filters plus row-level security keyed on a
  transaction-local ``app.tenant_id`` setting (see ``schema.sql``).

Requires the ``postgres`` extra: ``pip install 'safe-rag-reference[postgres]'``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np

from saferag.embeddings import Vector
from saferag.models import Chunk, Scope, ScoredChunk, Trust
from saferag.text import STOPWORDS, tokens

if TYPE_CHECKING:
    import psycopg

_COLUMNS = (
    "chunk_id, doc_id, tenant_id, allowed_roles, trust, title, version, section, "
    "section_slug, ordinal, body, content_hash"
)


def schema_sql() -> str:
    return resources.files("saferag.store").joinpath("schema.sql").read_text("utf-8")


def _row_to_chunk(row: Sequence[Any]) -> Chunk:
    return Chunk(
        chunk_id=row[0],
        doc_id=row[1],
        tenant_id=row[2],
        allowed_roles=frozenset(row[3]),
        trust=Trust(row[4]),
        title=row[5],
        version=row[6],
        section=row[7],
        section_slug=row[8],
        ordinal=row[9],
        text=row[10],
        content_hash=row[11],
    )


def or_tsquery(query: str) -> str:
    """Build an OR tsquery from safe alphanumeric terms only.

    ``plainto_tsquery`` ANDs every word, which is too strict for natural-language
    questions. Terms are reduced to ``[a-z0-9]+`` so user input can never inject
    tsquery operators. The value is still passed as a bound parameter.
    """
    terms: list[str] = []
    for tok in tokens(query):
        if tok in STOPWORDS:
            continue
        terms.extend(re.findall(r"[a-z0-9]+", tok))
    return " | ".join(dict.fromkeys(terms))


class PgVectorStore:
    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        from pgvector.psycopg import register_vector

        self._conn = conn
        register_vector(conn)
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        conn.commit()
        version = tuple(int(p) for p in str(row[0]).split(".")[:2]) if row else (0, 0)
        # pgvector >= 0.8 can keep scanning the HNSW graph until enough rows pass
        # the filter, which matters when a tenant is a small slice of the index.
        self._iterative_scan = version >= (0, 8)

    def _scoped(self, tenant_id: str) -> None:
        self._conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))

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
        with self._conn.transaction():
            self._scoped(tenant_id)
            self._conn.execute(
                "DELETE FROM rag_chunks WHERE tenant_id = %s AND doc_id = %s", (tenant_id, doc_id)
            )
            with self._conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO rag_chunks ({_COLUMNS}, embedding_model, embedding) "  # noqa: S608
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        (
                            c.chunk_id,
                            c.doc_id,
                            c.tenant_id,
                            sorted(c.allowed_roles),
                            c.trust.value,
                            c.title,
                            c.version,
                            c.section,
                            c.section_slug,
                            c.ordinal,
                            c.text,
                            c.content_hash,
                            model_id,
                            np.asarray(v, dtype=np.float32),
                        )
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
        return len(chunks)

    def delete_document(self, tenant_id: str, doc_id: str) -> int:
        with self._conn.transaction():
            self._scoped(tenant_id)
            cur = self._conn.execute(
                "DELETE FROM rag_chunks WHERE tenant_id = %s AND doc_id = %s", (tenant_id, doc_id)
            )
            return cur.rowcount

    def indexed_model_ids(self, scope: Scope) -> set[str]:
        with self._conn.transaction():
            self._scoped(scope.tenant_id)
            rows = self._conn.execute(
                "SELECT DISTINCT embedding_model FROM rag_chunks "
                "WHERE tenant_id = %s AND allowed_roles && %s",
                (scope.tenant_id, sorted(scope.roles)),
            ).fetchall()
        return {r[0] for r in rows}

    def vector_search(self, scope: Scope, vector: Vector, k: int) -> list[ScoredChunk]:
        vec = np.asarray(vector, dtype=np.float32)
        with self._conn.transaction():
            self._scoped(scope.tenant_id)
            if self._iterative_scan:
                self._conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = self._conn.execute(
                f"SELECT {_COLUMNS}, 1 - (embedding <=> %s) AS sim FROM rag_chunks "  # noqa: S608
                "WHERE tenant_id = %s AND allowed_roles && %s "
                "ORDER BY embedding <=> %s, chunk_id LIMIT %s",
                (vec, scope.tenant_id, sorted(scope.roles), vec, k),
            ).fetchall()
        return [ScoredChunk(_row_to_chunk(r), float(r[-1]), {"vector": float(r[-1])}) for r in rows]

    def keyword_search(self, scope: Scope, query: str, k: int) -> list[ScoredChunk]:
        tsq = or_tsquery(query)
        if not tsq:
            return []
        with self._conn.transaction():
            self._scoped(scope.tenant_id)
            rows = self._conn.execute(
                f"SELECT {_COLUMNS}, ts_rank_cd(search_tsv, q) AS rank "  # noqa: S608
                "FROM rag_chunks, to_tsquery('english', %s) AS q "
                "WHERE tenant_id = %s AND allowed_roles && %s AND search_tsv @@ q "
                "ORDER BY rank DESC, chunk_id LIMIT %s",
                (tsq, scope.tenant_id, sorted(scope.roles), k),
            ).fetchall()
        return [
            ScoredChunk(_row_to_chunk(r), float(r[-1]), {"keyword": float(r[-1])}) for r in rows
        ]
