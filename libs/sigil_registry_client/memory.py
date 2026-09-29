"""In-process registry. The reference semantics for §12, used by tests, dev serving, and
lifecycle simulation. ``postgres.PostgresRegistry`` implements the same methods.

[FIXED] §12.1: the registry is the sole source of truth for what a document is and
whether it exists. The trie is derived from ``snapshot_ids`` and never edited directly.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable, Sequence
from datetime import timedelta

from sigil_core.ids import SemanticId
from sigil_identifiers.assignment import PrefixCounter
from sigil_registry_client.bloom import Bloom
from sigil_registry_client.records import SERVABLE, Alias, DocRecord, DocState, Resolved, now

ALIAS_TTL = timedelta(days=30)


class MemoryRegistry:
    def __init__(self, id_schema: str = "ids_v1"):
        self.id_schema = id_schema
        self.epoch = 0
        self._docs: dict[str, DocRecord] = {}
        self._by_sid: dict[SemanticId, str] = {}
        self._by_hash: dict[tuple[str, str], str] = {}
        self._aliases: dict[SemanticId, Alias] = {}
        self._history: list[tuple[str, SemanticId, object, object]] = []
        self._counter = PrefixCounter()
        self._lock = threading.RLock()
        self.audit: list[dict] = []  # §27.2 append-only mutation log

    def _log(self, actor: str, op: str, doc_uid: str, before: str | None, after: str | None) -> None:
        self.audit.append({"at": now().isoformat(), "actor": actor, "op": op, "doc_uid": doc_uid,
                           "before": before, "after": after, "epoch": self.epoch})

    # -- writes ---------------------------------------------------------------------------

    def create(self, *, tenant_id: str, codes: Sequence[int], content_hash: str, content_uri: str = "",
               title: str | None = None, rerank_snippet: str | None = None, parent_doc_uid: str | None = None,
               acl: dict | None = None, metadata: dict | None = None, content_trust: str = "trusted",
               near_dup_group: str | None = None, state: DocState = DocState.ACTIVE_COLD_START,
               doc_uid: str | None = None, actor: str = "ingestion") -> DocRecord:
        """Idempotent by (tenant, content hash): re-ingesting identical content returns the
        existing record and allocates nothing."""
        with self._lock:
            existing = self._by_hash.get((tenant_id, content_hash))
            if existing and self._docs[existing].state != DocState.TOMBSTONED:
                return self._docs[existing]
            sid = self._counter.allocate(codes)
            self.epoch += 1
            rec = DocRecord(
                doc_uid=doc_uid or str(uuid.uuid4()), tenant_id=tenant_id, semantic_id=sid,
                id_schema_version=self.id_schema, content_hash=content_hash, content_uri=content_uri,
                title=title, rerank_snippet=rerank_snippet, state=state, corpus_epoch=self.epoch,
                acl=acl or {}, metadata=metadata or {}, parent_doc_uid=parent_doc_uid,
                near_dup_group=near_dup_group, content_trust=content_trust,
            )
            self._docs[rec.doc_uid] = rec
            self._by_sid[sid] = rec.doc_uid
            self._by_hash[(tenant_id, content_hash)] = rec.doc_uid
            self._history.append((rec.doc_uid, sid, rec.created_at, None))
            self._log(actor, "create", rec.doc_uid, None, str(sid))
            return rec

    def update_content(self, doc_uid: str, *, content_hash: str, rerank_snippet: str | None,
                       new_codes: Sequence[int] | None, content_uri: str | None = None,
                       actor: str = "ingestion") -> DocRecord:
        """``new_codes`` is None when the hysteresis policy kept the identifier. Otherwise a
        new identifier is allocated and the old one becomes an alias for 30 days, never a
        dangling path (§5.3)."""
        with self._lock:
            rec = self._docs[doc_uid]
            self.epoch += 1
            changes = dict(content_hash=content_hash, rerank_snippet=rerank_snippet, corpus_epoch=self.epoch,
                           content_uri=content_uri or rec.content_uri)
            if new_codes is not None and tuple(new_codes) != rec.semantic_id.codes:
                sid = self._counter.allocate(new_codes)
                self._aliases[rec.semantic_id] = Alias(rec.semantic_id, doc_uid, "CONTENT_DRIFT", now() + ALIAS_TTL)
                self._by_sid[sid] = doc_uid
                self._history.append((doc_uid, sid, now(), None))
                changes.update(semantic_id=sid, state=DocState.ACTIVE_COLD_START)
            self._by_hash.pop((rec.tenant_id, rec.content_hash), None)
            self._by_hash[(rec.tenant_id, content_hash)] = doc_uid
            new = rec.evolve(**changes)
            self._docs[doc_uid] = new
            self._log(actor, "update_content", doc_uid, str(rec.semantic_id), str(new.semantic_id))
            return new

    def patch(self, doc_uid: str, *, title: str | None = None, metadata: dict | None = None,
              acl: dict | None = None, actor: str = "api") -> DocRecord:
        """Metadata only. Never re-identifies, never touches the trie (§15.3)."""
        with self._lock:
            rec = self._docs[doc_uid]
            new = rec.evolve(title=title if title is not None else rec.title,
                             metadata={**rec.metadata, **(metadata or {})},
                             acl=acl if acl is not None else rec.acl)
            self._docs[doc_uid] = new
            self._log(actor, "patch", doc_uid, None, None)
            return new

    def tombstone(self, doc_uid: str, actor: str = "api") -> DocRecord:
        with self._lock:
            rec = self._docs[doc_uid]
            self.epoch += 1
            new = rec.evolve(state=DocState.TOMBSTONED, deleted_at=now(), corpus_epoch=self.epoch)
            self._docs[doc_uid] = new
            self._log(actor, "tombstone", doc_uid, str(rec.semantic_id), None)
            return new

    def hard_delete(self, doc_uid: str, actor: str) -> None:
        """Erasure (§27.2): content pointer dropped, row anonymized, identifier permanently
        retired. The counter never re-issues its ordinal."""
        with self._lock:
            rec = self.tombstone(doc_uid, actor)
            self._docs[doc_uid] = rec.evolve(content_uri="", title=None, rerank_snippet=None, metadata={}, acl={})
            self._log(actor, "hard_delete", doc_uid, None, None)

    def set_state(self, doc_uids: Iterable[str], state: DocState) -> None:
        with self._lock:
            for u in doc_uids:
                if self._docs[u].state != DocState.TOMBSTONED:
                    self._docs[u] = self._docs[u].evolve(state=state)

    def restore(self, records: Iterable[DocRecord]) -> MemoryRegistry:
        """Load records exactly as exported (dev seeding, tests). Identifiers are kept, and the
        counters resume past them so nothing is re-issued."""
        with self._lock:
            for rec in records:
                self._docs[rec.doc_uid] = rec
                self._by_sid[rec.semantic_id] = rec.doc_uid
                if rec.state != DocState.TOMBSTONED:
                    self._by_hash[(rec.tenant_id, rec.content_hash)] = rec.doc_uid
                self._history.append((rec.doc_uid, rec.semantic_id, rec.created_at, None))
                self.epoch = max(self.epoch, rec.corpus_epoch)
            self._counter = PrefixCounter.from_issued(s for _, s, _, _ in self._history)
        return self

    # -- reads ----------------------------------------------------------------------------

    def get(self, doc_uid: str) -> DocRecord | None:
        return self._docs.get(doc_uid)

    def children(self, parent_doc_uid: str) -> list[DocRecord]:
        """Retrieval units of one parent document, in chunk order."""
        return sorted((r for r in self._docs.values() if r.parent_doc_uid == parent_doc_uid),
                      key=lambda r: r.metadata.get("chunk", 0))

    def resolve(self, sids: Iterable[SemanticId], fresh: bool = False) -> dict[SemanticId, Resolved]:
        """Batch lookup, the hot path (§12.3). Missing identifiers are simply absent;
        expired aliases do not resolve."""
        out: dict[SemanticId, Resolved] = {}
        t = now()
        for sid in sids:
            uid = self._by_sid.get(sid)
            if uid is not None and self._docs[uid].semantic_id == sid:
                out[sid] = Resolved(self._docs[uid])
            elif (a := self._aliases.get(sid)) is not None and a.expires_at > t:
                out[sid] = Resolved(self._docs[a.doc_uid], via_alias=True)
        return out

    def snapshot_ids(self) -> list[SemanticId]:
        """Every live identifier: what the next trie snapshot must contain exactly."""
        return sorted(r.semantic_id for r in self._docs.values() if r.state in SERVABLE)

    def issued_ids(self) -> list[SemanticId]:
        return sorted({sid for _, sid, _, _ in self._history})

    def records(self, states: set[DocState] | None = None) -> list[DocRecord]:
        return [r for r in self._docs.values() if states is None or r.state in states]

    def tombstone_bloom(self) -> Bloom:
        dead = [r.doc_uid for r in self._docs.values() if r.state == DocState.TOMBSTONED]
        return Bloom(capacity=max(1000, 2 * len(dead))).update(dead)

    def hot_set_ratio(self) -> float:
        live = [r for r in self._docs.values() if r.state in SERVABLE]
        return sum(r.state == DocState.ACTIVE_COLD_START for r in live) / len(live) if live else 0.0
