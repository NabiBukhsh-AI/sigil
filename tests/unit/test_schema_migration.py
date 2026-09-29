"""The migration plan produces a consistent, verifiable set of artefacts; applying it against
a real database is covered in tests/integration/test_postgres_registry.py."""

import uuid

import pytest
from sigil_core.config import load_yaml
from sigil_core.ids import SemanticId
from sigil_trie import TrieSnapshot

from pipelines.schema_migration import CUTOVER, LOAD_COUNTERS, LOAD_MAP, _statements, apply_stage, plan
from tests.helpers import clustered_embeddings


def cfg16():
    cfg = load_yaml("configs/identifiers/ids_v1.yaml")
    cfg["schema_version"] = "ids_v2"
    cfg["identifiers"]["codes_per_level"] = 16
    cfg["quantizer"].update(kmeans_iters=10, kmeans_restarts=1)
    return cfg


def corpus(n):
    return [{"doc_uid": str(uuid.uuid5(uuid.NAMESPACE_URL, str(i))), "semantic_id": f"{i % 7}.0.0.0.0"} for i in range(n)]


def test_plan_artefacts_agree(tmp_path):
    emb, docs = clustered_embeddings(600), corpus(600)
    rep = plan(docs, emb, cfg16(), "cb_v2.0", tmp_path / "m")
    rows = [line.split(",") for line in (tmp_path / "m" / "mapping.csv").read_text().splitlines()]
    assert len(rows) == 600 and len({sid for _, sid in rows}) == 600
    with TrieSnapshot(rep["trie"], rep["trie_sha256"]) as t:
        assert t.id_schema == "ids_v2" and t.n_leaves == 600
        for _, hexsid in rows[:50]:
            assert SemanticId.unpack(bytes.fromhex(hexsid[2:])) in t
    counters = dict(line.split(",") for line in (tmp_path / "m" / "counters.csv").read_text().splitlines())
    assert sum(int(n) for n in counters.values()) == 600  # every ordinal accounted for
    assert rep["gates"]["l1_balance_ok"] and rep["refit"]


def test_final_replan_reuses_frozen_codebooks(tmp_path):
    emb, docs = clustered_embeddings(600), corpus(600)
    first = plan(docs, emb, cfg16(), "cb_v2.0", tmp_path / "a")
    final = plan(docs, emb, cfg16(), "cb_v2.0", tmp_path / "b", codebooks=tmp_path / "a" / "codebooks")
    assert not final["refit"] and final["codebook_sha256"] == first["codebook_sha256"]
    assert (tmp_path / "a" / "mapping.csv").read_text() == (tmp_path / "b" / "mapping.csv").read_text()


def test_cutover_is_set_based_and_two_person(tmp_path):
    stmts = _statements(CUTOVER)
    assert LOAD_MAP in stmts and LOAD_COUNTERS in stmts
    assert len(stmts) < 15  # a fixed handful of set-based statements, never one per document
    plan(corpus(200), clustered_embeddings(200), cfg16(), "cb_v2.0", tmp_path / "p")
    with pytest.raises(SystemExit, match="two different operators"):
        apply_stage("postgresql://unused", tmp_path / "p", "cutover", approved_by=["alice", "alice"])
