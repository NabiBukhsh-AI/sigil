"""PostgreSQL registry with an optional Redis read cache. Same contract as
``MemoryRegistry``; schema in ``services/registry/schema.sql``.

Hot path: one ``WHERE semantic_id = ANY(...)`` against a covering index, fronted by Redis
``MGET`` with a 300 s TTL. Writes delete the affected cache keys (CDC invalidation in
production; direct deletes here), and deletions additionally reach gateways through the
tombstone Bloom filter.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Sequence

from sigil_core.errors import PrefixCapacityExceeded
from sigil_core.ids import ORDINALS_PER_PREFIX, SemanticId
from sigil_registry_client.bloom import Bloom
from sigil_registry_client.memory import ALIAS_TTL
from sigil_registry_client.records import SERVABLE, DocRecord, DocState, Resolved, now

COLS = ("doc_uid, tenant_id, semantic_id, id_schema_version, content_hash, content_uri, title, rerank_snippet, "
        "state, corpus_epoch, acl, metadata, parent_doc_uid, near_dup_group, content_trust, created_at, "
        "updated_at, deleted_at")


def _rec(row) -> DocRecord:
    (uid, tenant, sid, schema, h, uri, title, snip, state, epoch, acl, meta, parent, group, trust,
     created, updated, deleted) = row
    return DocRecord(str(uid), tenant, SemanticId.unpack(bytes(sid)), schema, h, uri, title, snip,
                     DocState(state), epoch, acl, meta, str(parent) if parent else None, group, trust,
                     created, updated, deleted)


class PostgresRegistry:
    def __init__(self, dsn: str, id_schema: str = "ids_v1", redis_url: str | None = None, cache_ttl: int = 300):
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(dsn, min_size=1, max_size=16, open=True)
        self.id_schema = id_schema
        self.ttl = cache_ttl
        self.redis = None
        if redis_url:
            import redis

            self.redis = redis.Redis.from_url(redis_url)

    # -- helpers --------------------------------------------------------------------------

    def _key(self, sid: SemanticId) -> str:
        return f"sigil:{self.id_schema}:{sid}"

    def _invalidate(self, *sids: SemanticId) -> None:
        if self.redis and sids:
            self.redis.delete(*(self._key(s) for s in sids))

    @staticmethod
    def _audit(cur, actor, op, uid, before, after, epoch) -> None:
        cur.execute("INSERT INTO audit_log (actor, op, doc_uid, before_sid, after_sid, epoch) "
                    "VALUES (%s,%s,%s,%s,%s,%s)", (actor, op, uid, before, after, epoch))

    def _allocate(self, cur, codes: Sequence[int]) -> SemanticId:
        prefix = bytes(int(c) for c in codes)
        cur.execute(
            "INSERT INTO prefix_counters (id_schema_version, prefix, next_ordinal) VALUES (%s, %s, 1) "
            "ON CONFLICT (id_schema_version, prefix) DO UPDATE SET next_ordinal = prefix_counters.next_ordinal + 1 "
            "RETURNING next_ordinal - 1",
            (self.id_schema, prefix),
        )
        n = cur.fetchone()[0]
        if n >= ORDINALS_PER_PREFIX:
            raise PrefixCapacityExceeded(f"prefix {tuple(codes)} exhausted")
        return SemanticId.from_ordinal(tuple(codes), n)

    @property
    def epoch(self) -> int:
        with self.pool.connection() as c:
            return c.execute("SELECT last_value FROM corpus_epoch_seq").fetchone()[0]

    # -- writes ---------------------------------------------------------------------------

    def create(self, *, tenant_id, codes, content_hash, content_uri="", title=None, rerank_snippet=None,
               parent_doc_uid=None, acl=None, metadata=None, content_trust="trusted", near_dup_group=None,
               state=DocState.ACTIVE_COLD_START, doc_uid=None, actor="ingestion") -> DocRecord:
        with self.pool.connection() as c, c.transaction(), c.cursor() as cur:
            cur.execute(f"SELECT {COLS} FROM documents WHERE tenant_id=%s AND content_hash=%s "
                        "AND state <> 'TOMBSTONED'", (tenant_id, content_hash))
            if (row := cur.fetchone()) is not None:
                return _rec(row)
            sid = self._allocate(cur, codes)
            epoch = cur.execute("SELECT nextval('corpus_epoch_seq')").fetchone()[0]
            uid, t = doc_uid or str(uuid.uuid4()), now()
            cur.execute(
                f"INSERT INTO documents ({COLS}) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                f"RETURNING {COLS}",
                (uid, tenant_id, sid.pack(), self.id_schema, content_hash, content_uri, title, rerank_snippet,
                 state.value, epoch, json.dumps(acl or {}), json.dumps(metadata or {}), parent_doc_uid,
                 near_dup_group, content_trust, t, t, None),
            )
            rec = _rec(cur.fetchone())
            cur.execute("INSERT INTO identifier_history VALUES (%s,%s,%s,%s,NULL)", (uid, sid.pack(), self.id_schema, t))
            self._audit(cur, actor, "create", uid, None, str(sid), epoch)
        self._invalidate(sid)
        return rec

    def update_content(self, doc_uid, *, content_hash, rerank_snippet, new_codes, content_uri=None,
                       actor="ingestion") -> DocRecord:
        with self.pool.connection() as c, c.transaction(), c.cursor() as cur:
            old = _rec(cur.execute(f"SELECT {COLS} FROM documents WHERE doc_uid=%s FOR UPDATE", (doc_uid,)).fetchone())
            epoch = cur.execute("SELECT nextval('corpus_epoch_seq')").fetchone()[0]
            sid, state, t = old.semantic_id, old.state, now()
            if new_codes is not None and tuple(new_codes) != old.semantic_id.codes:
                sid, state = self._allocate(cur, new_codes), DocState.ACTIVE_COLD_START
                cur.execute("INSERT INTO identifier_aliases VALUES (%s,%s,%s,'CONTENT_DRIFT',%s,%s) "
                            "ON CONFLICT (id_schema_version, old_semantic_id) DO UPDATE SET doc_uid=EXCLUDED.doc_uid, "
                            "expires_at=EXCLUDED.expires_at",
                            (old.semantic_id.pack(), self.id_schema, doc_uid, t, t + ALIAS_TTL))
                cur.execute("UPDATE identifier_history SET valid_to=%s WHERE doc_uid=%s AND valid_to IS NULL", (t, doc_uid))
                cur.execute("INSERT INTO identifier_history VALUES (%s,%s,%s,%s,NULL)", (doc_uid, sid.pack(), self.id_schema, t))
            cur.execute(
                f"UPDATE documents SET content_hash=%s, rerank_snippet=%s, content_uri=%s, semantic_id=%s, state=%s, "
                f"corpus_epoch=%s, updated_at=%s WHERE doc_uid=%s RETURNING {COLS}",
                (content_hash, rerank_snippet, content_uri or old.content_uri, sid.pack(), state.value, epoch, t, doc_uid),
            )
            rec = _rec(cur.fetchone())
            self._audit(cur, actor, "update_content", doc_uid, str(old.semantic_id), str(sid), epoch)
        self._invalidate(old.semantic_id, sid)
        return rec

    def patch(self, doc_uid, *, title=None, metadata=None, acl=None, actor="api") -> DocRecord:
        with self.pool.connection() as c, c.transaction(), c.cursor() as cur:
            cur.execute(
                f"UPDATE documents SET title=COALESCE(%s, title), metadata = metadata || %s::jsonb, "
                f"acl=COALESCE(%s::jsonb, acl), updated_at=%s WHERE doc_uid=%s RETURNING {COLS}",
                (title, json.dumps(metadata or {}), json.dumps(acl) if acl is not None else None, now(), doc_uid),
            )
            rec = _rec(cur.fetchone())
            self._audit(cur, actor, "patch", doc_uid, None, None, rec.corpus_epoch)
        self._invalidate(rec.semantic_id)
        return rec

    def tombstone(self, doc_uid, actor="api") -> DocRecord:
        with self.pool.connection() as c, c.transaction(), c.cursor() as cur:
            epoch = cur.execute("SELECT nextval('corpus_epoch_seq')").fetchone()[0]
            cur.execute(f"UPDATE documents SET state='TOMBSTONED', deleted_at=%s, corpus_epoch=%s, updated_at=%s "
                        f"WHERE doc_uid=%s RETURNING {COLS}", (now(), epoch, now(), doc_uid))
            rec = _rec(cur.fetchone())
            self._audit(cur, actor, "tombstone", doc_uid, str(rec.semantic_id), None, epoch)
        self._invalidate(rec.semantic_id)
        return rec

    def hard_delete(self, doc_uid, actor) -> None:
        rec = self.tombstone(doc_uid, actor)
        with self.pool.connection() as c, c.transaction(), c.cursor() as cur:
            cur.execute("UPDATE documents SET content_uri='', title=NULL, rerank_snippet=NULL, metadata='{}', "
                        "acl='{}' WHERE doc_uid=%s", (doc_uid,))
            self._audit(cur, actor, "hard_delete", doc_uid, None, None, rec.corpus_epoch)

    def set_state(self, doc_uids: Iterable[str], state: DocState) -> None:
        with self.pool.connection() as c, c.transaction():
            c.execute("UPDATE documents SET state=%s, updated_at=%s WHERE doc_uid = ANY(%s) AND state <> 'TOMBSTONED'",
                      (state.value, now(), list(doc_uids)))

    # -- reads ----------------------------------------------------------------------------

    def get(self, doc_uid: str) -> DocRecord | None:
        with self.pool.connection() as c:
            row = c.execute(f"SELECT {COLS} FROM documents WHERE doc_uid=%s", (doc_uid,)).fetchone()
        return _rec(row) if row else None

    def children(self, parent_doc_uid: str) -> list[DocRecord]:
        with self.pool.connection() as c:
            rows = c.execute(f"SELECT {COLS} FROM documents WHERE parent_doc_uid=%s "
                             "ORDER BY (metadata->>'chunk')::int", (parent_doc_uid,)).fetchall()
        return [_rec(r) for r in rows]

    def resolve(self, sids: Iterable[SemanticId], fresh: bool = False) -> dict[SemanticId, Resolved]:
        """``fresh=True`` bypasses Redis: the final k are verified against the registry itself."""
        sids = list(dict.fromkeys(sids))
        out: dict[SemanticId, Resolved] = {}
        if self.redis and sids and not fresh:
            for sid, raw in zip(sids, self.redis.mget([self._key(s) for s in sids]), strict=True):
                if raw:
                    d = json.loads(raw)
                    out[sid] = Resolved(self._from_cache(d), d["via_alias"])
        missing = [s for s in sids if s not in out]
        if missing:
            packed = [s.pack() for s in missing]
            with self.pool.connection() as c:
                rows = c.execute(f"SELECT {COLS} FROM documents WHERE id_schema_version=%s AND semantic_id = ANY(%s) "
                                 "ORDER BY state = 'TOMBSTONED'", (self.id_schema, packed)).fetchall()
                found: dict[SemanticId, Resolved] = {}
                for r in rows:
                    rec = _rec(r)
                    found.setdefault(rec.semantic_id, Resolved(rec))
                aliased = [p for s, p in zip(missing, packed, strict=True) if s not in found]
                if aliased:
                    d_cols = ", ".join("d." + x.strip() for x in COLS.split(","))
                    rows = c.execute(
                        f"SELECT a.old_semantic_id, {d_cols} FROM identifier_aliases a JOIN documents d USING (doc_uid) "
                        "WHERE a.id_schema_version=%s AND a.old_semantic_id = ANY(%s) AND a.expires_at > now()",
                        (self.id_schema, aliased),
                    ).fetchall()
                    for row in rows:
                        found[SemanticId.unpack(bytes(row[0]))] = Resolved(_rec(row[1:]), True)
            out.update(found)
            if self.redis and found:
                pipe = self.redis.pipeline()
                for sid, res in found.items():
                    pipe.setex(self._key(sid), self.ttl, json.dumps(self._to_cache(res)))
                pipe.execute()
        return out

    @staticmethod
    def _to_cache(res: Resolved) -> dict:
        r = res.record
        return {"via_alias": res.via_alias, "doc_uid": r.doc_uid, "tenant_id": r.tenant_id, "sid": str(r.semantic_id),
                "schema": r.id_schema_version, "state": r.state.value, "title": r.title, "snippet": r.rerank_snippet,
                "acl": r.acl, "epoch": r.corpus_epoch, "uri": r.content_uri, "group": r.near_dup_group,
                "trust": r.content_trust, "metadata": r.metadata, "hash": r.content_hash}

    @staticmethod
    def _from_cache(d: dict) -> DocRecord:
        return DocRecord(d["doc_uid"], d["tenant_id"], SemanticId.parse(d["sid"]), d["schema"], d["hash"], d["uri"],
                         d["title"], d["snippet"], DocState(d["state"]), d["epoch"], d["acl"], d["metadata"],
                         near_dup_group=d["group"], content_trust=d["trust"])

    def snapshot_ids(self) -> list[SemanticId]:
        with self.pool.connection() as c:
            rows = c.execute("SELECT semantic_id FROM documents WHERE id_schema_version=%s AND state = ANY(%s)",
                             (self.id_schema, [s.value for s in SERVABLE])).fetchall()
        return sorted(SemanticId.unpack(bytes(r[0])) for r in rows)

    def issued_ids(self) -> list[SemanticId]:
        with self.pool.connection() as c:
            rows = c.execute("SELECT DISTINCT semantic_id FROM identifier_history WHERE id_schema_version=%s",
                             (self.id_schema,)).fetchall()
        return sorted(SemanticId.unpack(bytes(r[0])) for r in rows)

    def records(self, states: set[DocState] | None = None) -> list[DocRecord]:
        with self.pool.connection() as c:
            if states is None:
                rows = c.execute(f"SELECT {COLS} FROM documents").fetchall()
            else:
                rows = c.execute(f"SELECT {COLS} FROM documents WHERE state = ANY(%s)",
                                 ([s.value for s in states],)).fetchall()
        return [_rec(r) for r in rows]

    def tombstone_bloom(self) -> Bloom:
        with self.pool.connection() as c:
            dead = [str(r[0]) for r in c.execute("SELECT doc_uid FROM documents WHERE state='TOMBSTONED'").fetchall()]
        return Bloom(capacity=max(1000, 2 * len(dead))).update(dead)

    def hot_set_ratio(self) -> float:
        with self.pool.connection() as c:
            hot, live = c.execute(
                "SELECT count(*) FILTER (WHERE state='ACTIVE_COLD_START'), count(*) FROM documents WHERE state = ANY(%s)",
                ([s.value for s in SERVABLE],)).fetchone()
        return hot / live if live else 0.0
