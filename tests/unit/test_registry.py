"""§12 registry semantics on the reference implementation. PostgresRegistry runs the same
assertions in tests/integration against a live database."""

from datetime import timedelta

import pytest
from sigil_core.ids import SemanticId
from sigil_registry_client import MemoryRegistry, Principal
from sigil_registry_client.bloom import Bloom
from sigil_registry_client.records import DocState

CODES = (37, 210, 8, 155)


@pytest.fixture
def reg():
    return MemoryRegistry()


def make(reg, h="h1", tenant="t1", codes=CODES, **kw):
    return reg.create(tenant_id=tenant, codes=codes, content_hash=h, title="t", rerank_snippet="s", **kw)


def test_create_is_idempotent_by_content_hash(reg):
    a, b = make(reg), make(reg)
    assert a == b and len(reg.snapshot_ids()) == 1 and reg.epoch == 1


def test_ordinals_under_a_prefix(reg):
    recs = [make(reg, h=f"h{i}") for i in range(3)]
    assert [r.semantic_id.u for r in recs] == [0, 1, 2]
    assert reg.resolve([SemanticId.parse("37.210.8.155.1")])[SemanticId.parse("37.210.8.155.1")].record == recs[1]


def test_reidentification_leaves_an_alias(reg):
    old = make(reg)
    new = reg.update_content(old.doc_uid, content_hash="h2", rerank_snippet="s2", new_codes=(1, 2, 3, 4))
    assert new.semantic_id == SemanticId.parse("1.2.3.4.0") and new.state == DocState.ACTIVE_COLD_START
    hit = reg.resolve([old.semantic_id])[old.semantic_id]
    assert hit.via_alias and hit.record.doc_uid == old.doc_uid
    assert reg.snapshot_ids() == [new.semantic_id]  # the trie follows the new path


def test_alias_expires(reg):
    old = make(reg)
    reg.update_content(old.doc_uid, content_hash="h2", rerank_snippet=None, new_codes=(1, 2, 3, 4))
    a = reg._aliases[old.semantic_id]
    reg._aliases[old.semantic_id] = type(a)(a.old, a.doc_uid, a.reason, a.expires_at - timedelta(days=31))
    assert reg.resolve([old.semantic_id]) == {}


def test_metadata_patch_never_reidentifies(reg):
    r = make(reg)
    p = reg.patch(r.doc_uid, metadata={"species": "canine"})
    assert p.semantic_id == r.semantic_id and p.metadata == {"species": "canine"} and reg.epoch == 1


def test_tombstone_and_erasure(reg):
    r = make(reg)
    reg.tombstone(r.doc_uid)
    assert reg.snapshot_ids() == [] and r.doc_uid in reg.tombstone_bloom()
    reg.hard_delete(r.doc_uid, actor="dpo")
    gone = reg.get(r.doc_uid)
    assert gone.title is None and gone.content_uri == "" and gone.state == DocState.TOMBSTONED
    again = make(reg, h="h1")  # same content re-ingested: new document, never the retired id
    assert again.semantic_id.u == 1
    assert [e["op"] for e in reg.audit] == ["create", "tombstone", "tombstone", "hard_delete", "create"]


def test_acl_and_tenant_isolation(reg):
    r = make(reg, acl={"roles": ["vet"]})
    assert r.allows(Principal("t1", frozenset({"vet"})))
    assert not r.allows(Principal("t1", frozenset({"reception"})))
    assert not r.allows(Principal("t2", frozenset({"vet"})))


def test_bloom_wire_round_trip():
    b = Bloom(capacity=1000).update(f"d{i}" for i in range(500))
    c = Bloom.from_wire(b.to_wire())
    assert all(f"d{i}" in c for i in range(500))
    assert sum(f"x{i}" in c for i in range(10000)) < 20
