"""Integration tests against PostgreSQL + pgvector.

Run with ``SAFERAG_TEST_DATABASE_URL=postgresql://postgres@localhost/saferag``
(an admin connection used only to apply the schema and create a least-privilege
application role). Skipped when the variable is not set. CI runs these against
a pgvector service container; no external network or paid API is involved.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg import conninfo, sql  # noqa: E402

from saferag.config import Settings  # noqa: E402
from saferag.embeddings import HashingEmbedder  # noqa: E402
from saferag.indexing import build_pipeline, index_documents  # noqa: E402
from saferag.models import AnswerStatus, Document  # noqa: E402
from saferag.store.postgres import PgVectorStore, or_tsquery, schema_sql  # noqa: E402

from .conftest import LARKSPUR_EMPLOYEE, LARKSPUR_SECURITY, QUILLFEATHER_EMPLOYEE  # noqa: E402

pytestmark = pytest.mark.postgres
ADMIN_URL = os.environ.get("SAFERAG_TEST_DATABASE_URL")
APP_ROLE = "saferag_app_test"
APP_PASSWORD = "local-test-only"  # throwaway role in a throwaway database

if not ADMIN_URL:
    pytest.skip("SAFERAG_TEST_DATABASE_URL not set", allow_module_level=True)


@pytest.fixture(scope="module")
def app_conn(docs: list[Document]) -> Iterator[Any]:
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute("DROP TABLE IF EXISTS rag_chunks")
        admin.execute(schema_sql())
        exists = admin.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (APP_ROLE,)).fetchone()
        if not exists:
            admin.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS").format(
                    sql.Identifier(APP_ROLE), sql.Literal(APP_PASSWORD)
                )
            )
        # Least privilege: no ownership, no DDL, no UPDATE.
        admin.execute(
            sql.SQL("GRANT SELECT, INSERT, DELETE ON rag_chunks TO {}").format(
                sql.Identifier(APP_ROLE)
            )
        )
    url = conninfo.make_conninfo(ADMIN_URL, user=APP_ROLE, password=APP_PASSWORD)
    with psycopg.connect(url) as conn:
        index_documents(docs, PgVectorStore(conn), HashingEmbedder(), Settings())
        yield conn


def test_or_tsquery_strips_operators() -> None:
    assert or_tsquery("meal & limit | !drop:* <-> (x)") == "meal | limit | drop | x"
    assert or_tsquery("the and of") == ""


def test_scoped_search(app_conn: Any) -> None:
    store = PgVectorStore(app_conn)
    [vec] = HashingEmbedder().embed(["daily meal limit"])
    for scope in (LARKSPUR_EMPLOYEE, QUILLFEATHER_EMPLOYEE):
        for hit in store.vector_search(scope, vec, 20) + store.keyword_search(scope, "meal", 20):
            assert hit.chunk.tenant_id == scope.tenant_id
            assert scope.roles & hit.chunk.allowed_roles


def test_role_filter(app_conn: Any) -> None:
    store = PgVectorStore(app_conn)
    emp = store.keyword_search(LARKSPUR_EMPLOYEE, "incident bridge code", 20)
    sec = store.keyword_search(LARKSPUR_SECURITY, "incident bridge code", 20)
    assert all(h.chunk.doc_id != "larkspur-incident-runbook" for h in emp)
    assert any(h.chunk.doc_id == "larkspur-incident-runbook" for h in sec)


def test_rls_blocks_unscoped_and_cross_tenant_reads(app_conn: Any) -> None:
    # No tenant set: the policy matches nothing even with no WHERE clause.
    assert app_conn.execute("SELECT count(*) FROM rag_chunks").fetchone()[0] == 0
    app_conn.rollback()
    with app_conn.transaction():
        app_conn.execute("SELECT set_config('app.tenant_id', 'larkspur', true)")
        tenants = app_conn.execute("SELECT DISTINCT tenant_id FROM rag_chunks").fetchall()
        assert tenants == [("larkspur",)]
        # Even an explicit request for another tenant's rows returns nothing.
        n = app_conn.execute(
            "SELECT count(*) FROM rag_chunks WHERE tenant_id = 'quillfeather'"
        ).fetchone()[0]
        assert n == 0


def test_rls_blocks_cross_tenant_writes(app_conn: Any) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege), app_conn.transaction():
        app_conn.execute("SELECT set_config('app.tenant_id', 'larkspur', true)")
        app_conn.execute(
            "INSERT INTO rag_chunks (tenant_id, chunk_id, doc_id, allowed_roles, trust, title, "
            "version, section, section_slug, ordinal, body, content_hash, embedding_model, "
            "embedding) VALUES ('quillfeather', 'x', 'x', '{employee}', 'authoritative', 't', "
            "'1', 's', 's', 0, 'b', 'h', 'm', array_fill(0, ARRAY[384])::vector)"
        )


def test_pipeline_on_postgres(app_conn: Any) -> None:
    pipeline = build_pipeline(store=PgVectorStore(app_conn))
    ok = pipeline.answer(LARKSPUR_EMPLOYEE, "What is the daily meal limit for domestic travel?")
    assert ok.status is AnswerStatus.ANSWERED and "$75" in ok.answer
    cross = pipeline.answer(LARKSPUR_EMPLOYEE, "What is form QL-EXP-2 used for?")
    assert cross.status is AnswerStatus.REFUSED
    assert all(c.startswith("larkspur-") for c in cross.trace.retrieved_ids)


def test_reindex_is_idempotent(app_conn: Any, docs: list[Document]) -> None:
    store = PgVectorStore(app_conn)
    first = index_documents(docs, store, HashingEmbedder(), Settings())
    second = index_documents(docs, store, HashingEmbedder(), Settings())
    assert first.chunks == second.chunks
    assert store.indexed_model_ids(LARKSPUR_EMPLOYEE) == {"hashing-v1-d384"}
