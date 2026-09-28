"""Registry record types. §12.2."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum

from sigil_core.ids import SemanticId


class DocState(StrEnum):
    PENDING = "PENDING"
    ACTIVE_COLD_START = "ACTIVE_COLD_START"  # in the trie, not yet reinforced by training
    ACTIVE_LEARNED = "ACTIVE_LEARNED"
    TOMBSTONED = "TOMBSTONED"
    LOW_COVERAGE = "LOW_COVERAGE"  # permanently routed to the lexical channel (§7.4)
    QUARANTINED = "QUARANTINED"  # untrusted source awaiting review (§27)


SERVABLE = {DocState.ACTIVE_COLD_START, DocState.ACTIVE_LEARNED, DocState.LOW_COVERAGE}


def now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class DocRecord:
    doc_uid: str
    tenant_id: str
    semantic_id: SemanticId
    id_schema_version: str
    content_hash: str
    content_uri: str
    title: str | None
    rerank_snippet: str | None
    state: DocState
    corpus_epoch: int
    acl: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    parent_doc_uid: str | None = None
    near_dup_group: str | None = None
    content_trust: str = "trusted"  # §27: provenance signal handed to consumers
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)
    deleted_at: datetime | None = None

    def allows(self, p: Principal) -> bool:
        """Tenant must match exactly; if the ACL names roles, the principal needs one."""
        if self.tenant_id != p.tenant_id:
            return False
        roles = set(self.acl.get("roles", []))
        return not roles or bool(roles & p.roles)

    def evolve(self, **kw) -> DocRecord:
        return replace(self, updated_at=now(), **kw)


@dataclass(frozen=True)
class Alias:
    old: SemanticId
    doc_uid: str
    reason: str  # CONTENT_DRIFT | MERGE | SCHEMA_MIGRATION
    expires_at: datetime


@dataclass(frozen=True)
class Resolved:
    record: DocRecord
    via_alias: bool = False
