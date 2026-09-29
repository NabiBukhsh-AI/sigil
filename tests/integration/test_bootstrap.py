"""A bootstrapped deployment is verifiable and serves through the router before any model is
trained; the registry seed restores identifiers exactly."""

import json

from fastapi.testclient import TestClient
from sigil_registry_client.http import record_from_wire

from scripts.bootstrap_corpus import bootstrap
from scripts.local_stack import synthetic_corpus
from scripts.verify_bundle import verify


def test_bootstrap_bundle_serves_via_router(tmp_path, monkeypatch):
    docs = [{"tenant_id": "dev", "title": f"Doc {i}", "text": t} for _, i, t in synthetic_corpus(8, 10)]
    out = bootstrap(docs, tmp_path, k=16, key=b"k")
    assert verify(out["bundle"], b"k") == []

    monkeypatch.setenv("SIGIL_BUNDLE", out["bundle"])
    monkeypatch.setenv("SIGIL_BUNDLE_KEY", "k")
    from services.generative_retrieval.main import from_bundle

    c = TestClient(from_bundle())
    assert c.get("/ready").json()["ready"]
    r = c.post("/retrieve", json={"query": "unique5a unique5b topic5word1"}).json()
    sids = {x["semantic_id"] for x in r["candidates"]}
    records = [record_from_wire(json.loads(x)) for x in (tmp_path / "registry.jsonl").read_text().splitlines()]
    assert str(next(x.semantic_id for x in records if x.title == "Doc 5")) in sids


def test_registry_seed_restores_and_never_reissues(tmp_path, monkeypatch):
    docs = [{"tenant_id": "dev", "title": f"Doc {i}", "text": t} for _, i, t in synthetic_corpus(4, 5)]
    bootstrap(docs, tmp_path, k=8)
    monkeypatch.setenv("SIGIL_REGISTRY_SEED", str(tmp_path / "registry.jsonl"))
    from services.registry.main import make_registry

    reg = make_registry({"registry": {"backend": "memory"}})
    seeded = [record_from_wire(json.loads(x)) for x in (tmp_path / "registry.jsonl").read_text().splitlines()]
    assert sorted(reg.snapshot_ids()) == sorted(r.semantic_id for r in seeded)
    busiest = max(seeded, key=lambda r: r.semantic_id.ordinal)
    new = reg.create(tenant_id="dev", codes=busiest.semantic_id.codes, content_hash="fresh")
    assert new.semantic_id.ordinal == busiest.semantic_id.ordinal + 1
