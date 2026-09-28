"""§33 integration row, in-process: every service wired locally, no external dependencies.

Ingest -> trie snapshot -> retrieve through the gateway; a new document is live in the hot
channel immediately and in the generative channel after the next snapshot; a deletion
disappears as soon as the tombstone filter refreshes; tenants never see each other's
documents; reranker and GPU outages degrade instead of failing.
"""

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sigil_core.telemetry import MissReason
from sigil_decoding import DecodeConfig
from sigil_eval.baselines.bm25 import BM25
from sigil_identifiers.quantizer import RQKMeans
from sigil_registry_client import MemoryRegistry
from sigil_registry_client.records import DocState, Principal

from services.gateway.main import Gateway, GatewayConfig, create_app, local_backends
from services.gateway.schemas import ExplainRequest, RetrieveRequest
from services.generative_retrieval.engine import Engine
from services.generative_retrieval.snapshot_manager import SnapshotManager
from services.ingestion.main import ContentStore, Pipeline
from services.trie_builder.main import build_snapshot
from tests.helpers import RouterScorer, hash_embed, overlap_reranker

TOPICS, PER_TOPIC = 16, 10
P = Principal("clinic")


def doc_text(rng, t, i):
    topic = [f"topic{t}word{j}" for j in range(12)]
    return " ".join(list(rng.choice(topic, 30)) + [f"unique{i}a", f"unique{i}b", f"unique{i}c"])


class World:
    def __init__(self, tmp):
        rng = np.random.default_rng(0)
        self.docs = [(t, i, doc_text(rng, t, i)) for i in range(TOPICS * PER_TOPIC) for t in [i % TOPICS]]
        q = RQKMeans(k=16, iters=10, restarts=1).fit(hash_embed([d for _, _, d in self.docs]))
        self.reg = MemoryRegistry()
        self.hot = BM25()
        self.full = BM25()
        self.tmp = tmp
        self.pipe = Pipeline(self.reg, q, hash_embed, self.hot, ContentStore(tmp / "content"))
        self.parent = {}
        for t, i, text in self.docs:
            r = self.pipe.add(tenant_id="clinic", title=f"Doc {i}", content=text, initial_load=True)
            self.parent[i] = r["doc_uid"]
        self.snaps = SnapshotManager("ids_v1")
        self.rebuild()
        self.engine = Engine(RouterScorer(q), self.snaps)
        cfg = GatewayConfig(bundle_id="bundle_test", id_schema="ids_v1",
                            decode=DecodeConfig(beam=16, prefixes_expanded=16),
                            deadlines_ms={"generative": 10_000, "lexical": 10_000, "rerank": 10_000})
        self.backends = local_backends(self.engine, self.hot, self.full, self.reg, overlap_reranker)
        self.gw = Gateway(cfg, self.backends)

    def rebuild(self):
        info = build_snapshot(self.reg, self.tmp / "tries", "ids_v1")
        self.snaps.swap(info["path"], info["sha256"])
        self.full = BM25()
        live = {str(r.semantic_id): f"{r.title}\n{r.rerank_snippet}" for r in self.reg.records()
                if r.state not in (DocState.TOMBSTONED, DocState.QUARANTINED)}
        self.full.add(live.keys(), live.values())
        if hasattr(self, "backends"):
            self.backends.full_lexical = self.full.search
        return info

    def ask(self, query, tenant="clinic", k=10, principal=P, **opts):
        req = RetrieveRequest(query=query, tenant_id=tenant, k=k, options=opts or {})
        return self.gw.retrieve(req, principal)[0]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return World(tmp_path_factory.mktemp("world"))


def titles(resp):
    return [r["title"] for r in resp["results"]]


def test_semantic_queries_retrieve_their_documents(world):
    hits, attributed_a = 0, 0
    for t, i, _ in world.docs[::4]:
        resp = world.ask(f"unique{i}a unique{i}b topic{t}word1 topic{t}word2")
        assert resp["bundle_id"] == "bundle_test" and not resp["degraded_mode"]
        hits += f"Doc {i}" in titles(resp)
        attributed_a += sum(r["channel"] == "generative" for r in resp["results"])
    n = len(world.docs[::4])
    assert hits / n >= 0.8, hits / n
    assert attributed_a / (n * 10) >= 0.7  # §14.3 channel A result share


def test_new_document_hot_then_generative(world):
    r = world.pipe.add(tenant_id="clinic", title="Fresh protocol",
                       content="freshzeta freshomega freshkappa " + " ".join(f"topic3word{j}" for j in range(12)) * 2)
    resp = world.ask("freshzeta freshomega")
    got = [x for x in resp["results"] if x["title"] == "Fresh protocol"]
    assert got and got[0]["channel"] == "hot_lexical" and got[0]["state"] == "ACTIVE_COLD_START"
    world.rebuild()  # next trie snapshot: now reachable by generative fan-out (E7)
    gen = world.engine.retrieve("freshzeta freshomega topic3word1", replace(world.gw.cfg.decode, beam=32,
                                                                           prefixes_expanded=32))
    assert r["chunks"][0]["semantic_id"] in [c["semantic_id"] for c in gen["candidates"]]


def test_deletion_disappears_after_tombstone_refresh(world):
    i = 7
    assert "Doc 7" in titles(world.ask(f"unique{i}a unique{i}b unique{i}c"))
    world.pipe.delete(world.parent[i])
    world.gw.refresh_tombstones()
    assert "Doc 7" not in titles(world.ask(f"unique{i}a unique{i}b unique{i}c"))


def test_poisoned_corpus_never_crosses_tenants(world):
    world.pipe.add(tenant_id="rival", title="Rival secret", content="secretalpha secretbeta " * 20)
    world.rebuild()
    for q in ("secretalpha secretbeta", "secretalpha"):
        assert "Rival secret" not in titles(world.ask(q))
    assert "Rival secret" in titles(world.ask("secretalpha secretbeta", tenant="rival", principal=Principal("rival")))


def test_untrusted_source_is_quarantined(world):
    world.pipe.add(tenant_id="clinic", title="Injected", content="ignoreprevious instructions " * 20, trusted=False)
    world.rebuild()
    assert "Injected" not in titles(world.ask("ignoreprevious instructions"))


def test_reranker_outage_degrades_not_fails(world):
    def broken(q, texts):
        raise RuntimeError("reranker down")

    world.backends.rerank = broken
    try:
        resp = world.ask("unique3a unique3b topic3word1")
        assert resp["degraded_mode"] and resp["results"]
        assert all(r["scores"]["cross_encoder"] is None for r in resp["results"])
    finally:
        world.backends.rerank = overlap_reranker


def test_gpu_outage_falls_back_to_lexical_with_flag(world):
    good = world.backends.generative

    def down(*a, **k):
        raise ConnectionError("gpu pool unavailable")

    world.backends.generative = down
    try:
        resp = world.ask("unique5a unique5b")
        assert resp["degraded_mode"] and resp["channels"]["full_lexical"]["invoked"]
        assert "Doc 5" in titles(resp)
        assert "generative_error" in resp["channels"]["full_lexical"]["reasons"]
    finally:
        world.backends.generative = good
        world.gw.breaker.success()


def test_identifier_queries_route_lexical(world):
    resp = world.ask('"unique9a unique9b"')
    assert resp["channels"]["full_lexical"]["reasons"] == ["identifier_like_query"]
    assert resp["channels"]["generative"]["candidates"] == 0
    assert "Doc 9" in titles(resp)


def test_identical_requests_identical_results(world):
    a, b = world.ask("unique11a topic11word4"), world.ask("unique11a topic11word4")
    assert [r["doc_uid"] for r in a["results"]] == [r["doc_uid"] for r in b["results"]]


def test_explain_names_a_closed_set_reason(world):
    out = world.gw.explain(ExplainRequest(query="unique12a unique12b", tenant_id="clinic",
                                          expected=world.reg.children(world.parent[12])[0].doc_uid), P)
    assert out["oracle_check"]["reason"] in set(MissReason)
    assert out["beam_trace"][0]["level"] == 1


def test_http_auth_and_scopes(world):
    keys = {"r": {"tenant": "clinic", "scopes": ["retrieve"]}, "o": {"tenant": "clinic", "scopes": ["operator"]}}
    c = TestClient(create_app(world.gw, keys))
    body = {"query": "unique2a", "tenant_id": "clinic", "k": 5}
    assert c.post("/v1/retrieve", json=body).status_code == 401
    assert c.post("/v1/retrieve", json={**body, "tenant_id": "rival"}, headers={"Authorization": "Bearer r"}).status_code == 403
    ok = c.post("/v1/retrieve", json=body, headers={"Authorization": "Bearer r"})
    assert ok.status_code == 200 and ok.json()["trace_id"]
    ex = {"query": "unique2a", "tenant_id": "clinic"}
    assert c.post("/v1/debug/explain", json=ex, headers={"Authorization": "Bearer r"}).status_code == 403
    assert c.post("/v1/debug/explain", json=ex, headers={"Authorization": "Bearer o"}).status_code == 200
    capped = c.post("/v1/retrieve", json={**body, "options": {"beam_width": 5000}}, headers={"Authorization": "Bearer r"})
    assert capped.status_code == 200  # server-side cap overrides the client, no failure
