"""§27, §33 security row: write scopes and audit, erasure scope, server-side caps at the
generative service, provenance labels, and no cross-tenant results via any channel."""

import pytest
from fastapi.testclient import TestClient
from sigil_decoding import DecodeConfig
from sigil_registry_client.records import Principal

from scripts.local_stack import build, synthetic_corpus
from services.generative_retrieval.main import create_app as grs_app
from services.ingestion.main import create_app as ingest_app

KEYS = {
    "w": {"tenant": "clinic", "scopes": ["write"]},
    "r": {"tenant": "clinic", "scopes": ["retrieve"]},
    "rival": {"tenant": "rival", "scopes": ["write"]},
    "admin": {"tenant": "clinic", "scopes": ["write", "erase"]},
}
DOC = {"tenant_id": "clinic", "title": "Canine dosing", "content": "amoxicillin dosing canine twenty mg per kg " * 5}


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    s = build(synthetic_corpus(8, 6), tmp_path_factory.mktemp("sec"))
    yield s
    s.close()


def test_write_requires_scope_and_is_audited(stack):
    c = TestClient(ingest_app(stack.pipe, KEYS))
    assert c.post("/v1/documents", json=DOC).status_code == 403
    assert c.post("/v1/documents", json=DOC, headers={"Authorization": "Bearer r"}).status_code == 403
    assert c.post("/v1/documents", json=DOC, headers={"Authorization": "Bearer rival"}).status_code == 403
    r = c.post("/v1/documents", json=DOC, headers={"Authorization": "Bearer w"})
    assert r.status_code == 201
    uid = r.json()["doc_uid"]
    assert any(e["op"] == "create" for e in stack.reg.audit)
    # another tenant cannot modify or delete it
    assert c.delete(f"/v1/documents/{uid}", headers={"Authorization": "Bearer rival"}).status_code == 403
    # hard delete needs the elevated erase scope
    assert c.delete(f"/v1/documents/{uid}?hard=true", headers={"Authorization": "Bearer w"}).status_code == 403
    assert c.delete(f"/v1/documents/{uid}?hard=true", headers={"Authorization": "Bearer admin"}).status_code == 200
    assert any(e["op"] == "hard_delete" and e["actor"] == "clinic" for e in stack.reg.audit)


def test_generative_service_enforces_beam_cap(stack):
    c = TestClient(grs_app(stack.engine, "bundle_local", DecodeConfig(beam=16)))
    assert c.post("/retrieve", json={"query": "topic1word1", "beam": 16}).status_code == 200
    r = c.post("/retrieve", json={"query": "topic1word1", "beam": 100_000})
    assert r.status_code == 429 and r.headers["Retry-After"]
    assert c.post("/retrieve", json={"query": "x", "bundle_id": "other"}).status_code == 409


def test_corrupt_snapshot_is_refused_and_previous_kept(stack, tmp_path):
    c = TestClient(grs_app(stack.engine, "bundle_local"))
    before = c.get("/health").json()["trie_snapshot"]
    bad = tmp_path / "bad.trie"
    bad.write_bytes(b"SGLTRIE1" + b"\0" * 100)
    assert c.post("/snapshot", json={"path": str(bad), "sha256": "0" * 64}).status_code == 422
    assert c.get("/health").json()["trie_snapshot"] == before


def test_provenance_is_propagated(stack):
    stack.pipe.add(tenant_id="clinic", title="Vendor note", content="vendorzeta widget guidance " * 10)
    resp = stack.ask("vendorzeta widget")
    hit = [r for r in resp["results"] if r["title"] == "Vendor note"]
    assert hit and hit[0]["content_trust"] == "trusted"


def test_no_channel_leaks_other_tenants(stack):
    stack.pipe.add(tenant_id="rival", title="Rival plan", content="rivalomega merger terms " * 10)
    stack.rebuild()
    for q in ("rivalomega merger terms", '"rivalomega"'):  # generative + hot, then full lexical
        resp = stack.ask(q)
        assert all(r["title"] != "Rival plan" for r in resp["results"])
    assert any(e["event"] == "acl_drop" for e in stack.gw.audit)
    assert stack.ask("rivalomega merger", tenant="rival", principal=Principal("rival"))["results"]
