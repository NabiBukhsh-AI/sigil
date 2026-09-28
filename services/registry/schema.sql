-- SIGIL document registry. §12.2. PostgreSQL 16.
-- The registry is the sole source of truth for what a document is and whether it exists.

CREATE TYPE doc_state AS ENUM (
    'PENDING', 'ACTIVE_COLD_START', 'ACTIVE_LEARNED', 'TOMBSTONED', 'LOW_COVERAGE', 'QUARANTINED'
);

CREATE SEQUENCE corpus_epoch_seq;

-- "37.210.8.155.0" for humans, from the packed bytes.
CREATE FUNCTION sigil_sid_text(b BYTEA) RETURNS TEXT LANGUAGE sql IMMUTABLE STRICT AS $$
    SELECT string_agg(get_byte(b, i)::text, '.' ORDER BY i) FROM generate_series(0, length(b) - 1) AS i
$$;

CREATE TABLE documents (
    doc_uid           UUID PRIMARY KEY,
    tenant_id         TEXT NOT NULL,
    semantic_id       BYTEA NOT NULL,                  -- packed codes, 5 to 7 bytes
    semantic_id_text  TEXT GENERATED ALWAYS AS (sigil_sid_text(semantic_id)) STORED,
    id_schema_version TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    content_uri       TEXT NOT NULL,
    title             TEXT,
    rerank_snippet    TEXT,                            -- first 256 tokens; avoids a blob fetch
    parent_doc_uid    UUID,
    near_dup_group    TEXT,
    content_trust     TEXT NOT NULL DEFAULT 'trusted',
    state             doc_state NOT NULL,
    acl               JSONB NOT NULL,
    metadata          JSONB NOT NULL,
    corpus_epoch      BIGINT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL,
    updated_at        TIMESTAMPTZ NOT NULL,
    deleted_at        TIMESTAMPTZ
);
CREATE UNIQUE INDEX documents_sid ON documents (id_schema_version, semantic_id) WHERE state <> 'TOMBSTONED';
CREATE INDEX documents_sid_all ON documents (id_schema_version, semantic_id)
    INCLUDE (doc_uid, state, title, rerank_snippet, acl, tenant_id);  -- covering index for batch resolve
CREATE INDEX documents_tenant_state ON documents (tenant_id, state);
CREATE INDEX documents_cold_start ON documents (created_at) WHERE state = 'ACTIVE_COLD_START';
CREATE UNIQUE INDEX documents_hash ON documents (tenant_id, content_hash) WHERE state <> 'TOMBSTONED';

-- Per-prefix ordinal counter. The row lock taken by the upsert is the per-prefix
-- serialization §24 F3 asks for, scoped to the 4-level prefix rather than the table.
-- It only ever increases, so an identifier is never reused within a schema (§12.4).
CREATE TABLE prefix_counters (
    id_schema_version TEXT NOT NULL,
    prefix            BYTEA NOT NULL,
    next_ordinal      INTEGER NOT NULL,
    PRIMARY KEY (id_schema_version, prefix)
);

CREATE TABLE identifier_aliases (
    old_semantic_id   BYTEA,
    id_schema_version TEXT,
    doc_uid           UUID NOT NULL,
    reason            TEXT NOT NULL,                   -- CONTENT_DRIFT | MERGE | SCHEMA_MIGRATION
    created_at        TIMESTAMPTZ NOT NULL,
    expires_at        TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id_schema_version, old_semantic_id)
);

CREATE TABLE identifier_history (
    doc_uid UUID, semantic_id BYTEA, id_schema_version TEXT,
    valid_from TIMESTAMPTZ, valid_to TIMESTAMPTZ
);
CREATE INDEX identifier_history_sid ON identifier_history (id_schema_version, semantic_id);

-- §27.2 append-only audit log of every mutation.
CREATE TABLE audit_log (
    id          BIGSERIAL PRIMARY KEY,
    at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor       TEXT NOT NULL,
    op          TEXT NOT NULL,
    doc_uid     UUID,
    before_sid  TEXT,
    after_sid   TEXT,
    epoch       BIGINT
);
REVOKE UPDATE, DELETE ON audit_log FROM PUBLIC;
