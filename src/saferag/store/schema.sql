-- Schema for the PostgreSQL + pgvector store.
--
-- Isolation is enforced twice:
--   1. Every application query filters on tenant_id and allowed_roles (pre-filter).
--   2. Row-level security rejects rows whose tenant_id differs from the
--      transaction-local setting app.tenant_id. If application code forgets the
--      WHERE clause, the database still returns nothing from other tenants.
--
-- The application should connect as a dedicated non-owner, non-superuser role
-- (superusers and BYPASSRLS roles skip RLS). See tests/test_postgres.py for a
-- least-privilege role setup.
--
-- The embedding dimension is fixed here. Changing embedding models means a new
-- column or table plus a full re-index, never an in-place mix of vector spaces.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    tenant_id        text        NOT NULL,
    chunk_id         text        NOT NULL,
    doc_id           text        NOT NULL,
    allowed_roles    text[]      NOT NULL CHECK (cardinality(allowed_roles) > 0),
    trust            text        NOT NULL CHECK (trust IN ('authoritative', 'third_party')),
    title            text        NOT NULL,
    version          text        NOT NULL,
    section          text        NOT NULL,
    section_slug     text        NOT NULL,
    ordinal          integer     NOT NULL,
    body             text        NOT NULL,
    content_hash     text        NOT NULL,
    embedding_model  text        NOT NULL,
    embedding        vector(384) NOT NULL,
    search_tsv       tsvector GENERATED ALWAYS AS (
                         to_tsvector('english', title || ' ' || section || ' ' || body)
                     ) STORED,
    indexed_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS rag_chunks_tenant_doc ON rag_chunks (tenant_id, doc_id);
CREATE INDEX IF NOT EXISTS rag_chunks_roles ON rag_chunks USING gin (allowed_roles);
CREATE INDEX IF NOT EXISTS rag_chunks_tsv ON rag_chunks USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS rag_chunks_embedding
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);

ALTER TABLE rag_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_chunks FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON rag_chunks;
-- current_setting(..., true) returns NULL when unset, so an unscoped session
-- matches no rows: the policy fails closed.
CREATE POLICY tenant_isolation ON rag_chunks
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
