"""Client for services/registry. What the gateway, trie builder, and ingestion workers use
when the registry runs as its own service (§20.2 S4)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from sigil_core.ids import SemanticId
from sigil_registry_client.bloom import Bloom
from sigil_registry_client.records import DocRecord, DocState, Resolved


def record_to_wire(r: DocRecord) -> dict:
    return {
        "doc_uid": r.doc_uid, "tenant_id": r.tenant_id, "semantic_id": str(r.semantic_id),
        "id_schema_version": r.id_schema_version, "content_hash": r.content_hash, "content_uri": r.content_uri,
        "title": r.title, "rerank_snippet": r.rerank_snippet, "state": r.state.value, "corpus_epoch": r.corpus_epoch,
        "acl": r.acl, "metadata": r.metadata, "parent_doc_uid": r.parent_doc_uid, "near_dup_group": r.near_dup_group,
        "content_trust": r.content_trust, "created_at": r.created_at.isoformat(),
        "updated_at": r.updated_at.isoformat(), "deleted_at": r.deleted_at.isoformat() if r.deleted_at else None,
    }


def record_from_wire(d: dict) -> DocRecord:
    return DocRecord(
        d["doc_uid"], d["tenant_id"], SemanticId.parse(d["semantic_id"]), d["id_schema_version"], d["content_hash"],
        d["content_uri"], d["title"], d["rerank_snippet"], DocState(d["state"]), d["corpus_epoch"], d["acl"],
        d["metadata"], d.get("parent_doc_uid"), d.get("near_dup_group"), d.get("content_trust", "trusted"),
        datetime.fromisoformat(d["created_at"]), datetime.fromisoformat(d["updated_at"]),
        datetime.fromisoformat(d["deleted_at"]) if d.get("deleted_at") else None,
    )


class HttpRegistry:
    def __init__(self, base_url: str, timeout: float = 0.5):
        import httpx

        self.c = httpx.Client(base_url=base_url, timeout=timeout)

    def resolve(self, sids: Iterable[SemanticId], fresh: bool = False) -> dict[SemanticId, Resolved]:
        r = self.c.post("/internal/resolve", json={"semantic_ids": [str(s) for s in sids], "fresh": fresh})
        r.raise_for_status()
        return {SemanticId.parse(k): Resolved(record_from_wire(v["record"]), v["via_alias"])
                for k, v in r.json()["resolved"].items()}

    def snapshot_ids(self) -> list[SemanticId]:
        r = self.c.get("/internal/snapshot_ids", timeout=60)
        r.raise_for_status()
        return [SemanticId.parse(s) for s in r.json()["semantic_ids"]]

    def tombstone_bloom(self) -> Bloom:
        r = self.c.get("/internal/tombstones:bloom")
        r.raise_for_status()
        return Bloom.from_wire(r.json())

    @property
    def epoch(self) -> int:
        return self.c.get("/internal/epoch").json()["epoch"]
