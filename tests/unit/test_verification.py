"""Tier-1 verification (§13.1), including the dual-schema window of a migration (§15.6)."""

from dataclasses import replace

from sigil_core.ids import SemanticId
from sigil_registry_client import MemoryRegistry
from sigil_registry_client.bloom import Bloom
from sigil_registry_client.records import Principal, Resolved

from services.gateway.verification import Cand, tier1

P = Principal("t")


def setup():
    reg = MemoryRegistry()
    a = reg.create(tenant_id="t", codes=(1, 1, 1, 1), content_hash="a", metadata={"species": "canine"})
    b = reg.create(tenant_id="t", codes=(2, 2, 2, 2), content_hash="b", acl={"roles": ["vet"]})
    c = reg.create(tenant_id="other", codes=(3, 3, 3, 3), content_hash="c")
    return reg, a, b, c


def test_drops_each_failure_class_once():
    reg, a, b, c = setup()
    dead = reg.create(tenant_id="t", codes=(4, 4, 4, 4), content_hash="d")
    reg.tombstone(dead.doc_uid)
    missing = SemanticId.parse("9.9.9.9.0")
    cands = [Cand(s, "generative", gen=-1.0) for s in (a.semantic_id, b.semantic_id, c.semantic_id,
                                                       dead.semantic_id, missing)]
    v = tier1(cands, reg.resolve([x.sid for x in cands]), P, "ids_v1", Bloom(capacity=100))
    assert [k.record.doc_uid for k in v.kept] == [a.doc_uid]
    assert dict(v.dropped) == {"acl": 2, "tombstoned": 1, "missing": 1}  # b needs a role, c is another tenant
    assert v.skew == 1


def test_duplicate_documents_collapse_and_attribution_prefers_generative():
    reg, a, *_ = setup()
    cands = [Cand(a.semantic_id, "hot_lexical", bm25=3.0), Cand(a.semantic_id, "generative", gen=-2.0)]
    v = tier1(cands, reg.resolve([a.semantic_id]), P, "ids_v1", Bloom(capacity=100))
    assert len(v.kept) == 1 and v.kept[0].channel == "generative"
    assert v.kept[0].bm25 == 3.0 and v.kept[0].gen == -2.0


def test_metadata_filters():
    reg, a, *_ = setup()
    res = reg.resolve([a.semantic_id])
    assert tier1([Cand(a.semantic_id, "generative")], res, P, "ids_v1", Bloom(), {"metadata.species": ["canine"]}).kept
    assert not tier1([Cand(a.semantic_id, "generative")], res, P, "ids_v1", Bloom(), {"metadata.species": ["feline"]}).kept


def test_schema_mismatch_on_direct_hit_is_dropped_but_alias_hits_pass():
    reg, a, *_ = setup()
    v2_record = replace(a, id_schema_version="ids_v2")
    direct = {a.semantic_id: Resolved(v2_record)}
    assert tier1([Cand(a.semantic_id, "generative")], direct, P, "ids_v1", Bloom()).dropped["schema"] == 1
    # Mid-migration: an ids_v2 bundle resolves its identifier through a schema-scoped alias to a
    # record that still carries ids_v1. The hit is valid.
    staged = {a.semantic_id: Resolved(a, via_alias=True)}
    assert tier1([Cand(a.semantic_id, "generative")], staged, P, "ids_v2", Bloom()).kept
