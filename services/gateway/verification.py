"""Tier-1 verification. §13.1. Hard constraints, no model, runs on every candidate.

Any failure removes the candidate; the caller backfills from the next rank. ACL is checked
here, after retrieval, and again on the final k at response assembly, never only at query
time (§27 cross-tenant leakage).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from sigil_core.ids import SemanticId
from sigil_registry_client.records import SERVABLE, DocState, Principal, Resolved


@dataclass
class Cand:
    sid: SemanticId
    channel: str  # generative | hot_lexical | full_lexical
    gen: float | None = None  # generative sequence log-prob
    bm25: float | None = None
    record: object = None
    via_alias: bool = False
    ce: float | None = None
    final: float = 0.0


@dataclass
class Tier1:
    kept: list[Cand]
    dropped: Counter = field(default_factory=Counter)
    aliases: int = 0

    @property
    def skew(self) -> int:
        """Identifiers the trie produced that the registry does not know (§24 F2)."""
        return self.dropped["missing"]


def _passes_filters(metadata: dict, filters: Mapping[str, list[str]]) -> bool:
    for key, allowed in filters.items():
        v = metadata.get(key.removeprefix("metadata."))
        if not (set(v) if isinstance(v, list) else {v}) & set(allowed):
            return False
    return True


def tier1(cands: list[Cand], resolved: Mapping[SemanticId, Resolved], principal: Principal, id_schema: str,
          tombstones, filters: Mapping[str, list[str]] | None = None, audit: list | None = None) -> Tier1:
    out = Tier1(kept=[])
    by_doc: dict[str, Cand] = {}
    for c in cands:
        hit = resolved.get(c.sid)
        if hit is None:
            out.dropped["missing"] += 1
            continue
        rec = hit.record
        if rec.doc_uid in tombstones or rec.state == DocState.TOMBSTONED:
            out.dropped["tombstoned"] += 1
            continue
        if rec.state not in SERVABLE:
            out.dropped["not_servable"] += 1
            continue
        if not rec.allows(principal):
            out.dropped["acl"] += 1
            if audit is not None:
                audit.append({"event": "acl_drop", "tenant": principal.tenant_id, "doc_uid": rec.doc_uid})
            continue
        # Every registry lookup is scoped to the bundle's schema, so an alias hit already matched
        # an identifier of that schema. During a migration the record itself may carry the other
        # schema (§15.6: both identifier sets live at once); a direct hit never should.
        if not hit.via_alias and rec.id_schema_version != id_schema:
            out.dropped["schema"] += 1  # a deployment error, alarmed by the caller
            continue
        if filters and not _passes_filters(rec.metadata, filters):
            out.dropped["filtered"] += 1
            continue
        c.record, c.via_alias = rec, hit.via_alias
        out.aliases += hit.via_alias
        # One entry per document: channels and aliases can surface the same doc twice.
        prev = by_doc.get(rec.doc_uid)
        if prev is None:
            by_doc[rec.doc_uid] = c
        else:
            prev.gen = prev.gen if c.gen is None else c.gen if prev.gen is None else max(prev.gen, c.gen)
            prev.bm25 = prev.bm25 if c.bm25 is None else c.bm25 if prev.bm25 is None else max(prev.bm25, c.bm25)
            if prev.channel != "generative" and c.channel == "generative":
                prev.channel = "generative"  # attribution goes to channel A when both found it
            out.dropped["duplicate"] += 1
    out.kept = list(by_doc.values())
    return out
