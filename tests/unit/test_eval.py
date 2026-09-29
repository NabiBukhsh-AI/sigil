"""Phase 8: metric correctness, gates, baselines, and the brute-force oracle."""

import math

import numpy as np
import pytest
from sigil_core.ids import SemanticId
from sigil_decoding import DecodeConfig, decode
from sigil_eval import gates
from sigil_eval.baselines.bm25 import BM25, with_expansions
from sigil_eval.baselines.dense import DenseExact
from sigil_eval.baselines.hybrid import rrf
from sigil_eval.baselines.oracle import brute_force
from sigil_eval.golden_sets import held_out_slice
from sigil_eval.metrics import (
    ece,
    evaluate_run,
    kendall_tau,
    mrr_at,
    ndcg_at,
    prefix_survival,
    recall_at,
    survival_curve,
)
from sigil_trie import TrieSnapshot, write

from tests.helpers import ToyScorer


def test_metrics_against_hand_computed_values():
    rel = {"a": 2, "b": 1, "z": 1}
    ranked = ["x", "a", "b", "y"]
    assert recall_at(ranked, rel, 2) == pytest.approx(1 / 3)
    assert mrr_at(ranked, rel, 10) == 0.5
    dcg = 3 / math.log2(3) + 1 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at(ranked, rel, 10) == pytest.approx(dcg / idcg)
    m = evaluate_run({"q": ranked}, {"q": rel, "unjudged_run_missing": {"a": 1}})
    assert m["n_queries"] == 2 and m["hit@1"] == 0.0


def test_prefix_survival_finds_the_pruning_level():
    survivors = [np.array([[1], [2]]), np.array([[1, 5], [2, 6]]), np.array([[2, 6, 0]]), np.array([[2, 6, 0, 1]])]
    assert prefix_survival((1, 5, 9, 9), survivors) == 2
    assert prefix_survival((2, 6, 0, 1), survivors) == 4
    assert prefix_survival((3, 0, 0, 0), survivors) == 0
    assert survival_curve([0, 2, 4, 4]) == {
        "prefix_survival@1": 0.75, "prefix_survival@2": 0.75, "prefix_survival@3": 0.5, "prefix_survival@4": 0.5,
    }


def test_ece_and_kendall():
    assert ece(np.array([0.9] * 10), np.array([1] * 9 + [0])) == pytest.approx(0.0)
    assert kendall_tau(list("abcd"), list("abcd")) == 1.0
    assert kendall_tau(list("abcd"), list("dcba")) == -1.0


def passing():
    return {
        "valid_id_rate": 1.0, "ndcg@10": 0.50, "recall@10": 0.70, "mrr@10": 0.40,
        "held_out_doc_recall@10": 0.55, "ece": 0.03, "stale_id_rate": 0.0, "escape_rate": 0.001,
        "channel_c_query_share": 0.08, "channel_a_result_share": 0.85, "p95_latency_ms": 100.0,
        "cost_per_1k_queries": 1.0, "cross_tenant_leaks": 0,
    }


def test_gates_pass_and_block():
    assert gates.evaluate(passing(), passing()).passed
    bad = {**passing(), "valid_id_rate": 0.9999}
    v = gates.evaluate(bad, passing())
    assert not v.passed and [r.name for r in v.failures] == ["valid_id_rate"]
    # Improving NDCG by leaning on channel C is rejected (§14.3).
    creep = {**passing(), "ndcg@10": 0.60, "channel_c_query_share": 0.2}
    assert not gates.evaluate(creep, passing()).passed
    # Recall threshold is in points: 1.0 point = 0.01.
    assert gates.evaluate({**passing(), "recall@10": 0.691}, passing()).passed
    assert not gates.evaluate({**passing(), "recall@10": 0.689}, passing()).passed


def test_first_release_must_beat_the_baseline():
    # No incumbent and no baseline: relative gates cannot pass by being skipped.
    v = gates.evaluate(passing())
    assert {"ndcg@10", "recall@10", "mrr@10"} <= {r.name for r in v.failures}
    floor = {"ndcg@10": 0.45, "recall@10": 0.65, "mrr@10": 0.35, "p95_latency_ms": 200.0, "cost_per_1k_queries": 2.0}
    assert gates.evaluate(passing(), baseline=floor).passed
    weaker = {**floor, "recall@10": 0.705}  # zero tolerance against the floor
    assert "recall@10" in {r.name for r in gates.evaluate(passing(), baseline=weaker).failures}


def test_unmeasured_gate_fails_and_adapter_gates_apply():
    m = passing()
    del m["ece"]
    assert "ece" in [r.name for r in gates.evaluate(m, passing()).failures]
    v = gates.evaluate(passing(), passing(), kind=gates.ADAPTER)
    assert {r.name for r in v.failures} == {"old_doc_recall_regression", "cold_doc_recall"}


def test_offline_stage_checks_only_what_training_can_measure():
    offline = {k: v for k, v in passing().items()
               if k not in ("channel_c_query_share", "channel_a_result_share", "p95_latency_ms",
                            "cost_per_1k_queries", "cross_tenant_leaks")}
    assert gates.evaluate(offline, passing(), only=gates.OFFLINE).passed
    assert not gates.evaluate(offline, passing()).passed  # the release decision still needs everything


def test_bm25_ranks_and_tombstones():
    idx = BM25().add(["d1", "d2", "d3"], ["canine antibiotic course", "feline kidney disease", "canine diet"])
    assert idx.search("canine antibiotics course")[0][0] == "d1"
    idx.remove(["d1"])
    assert "d1" not in [k for k, _ in idx.search("canine course")]
    exp = with_expansions({"d2": "feline kidney"}, {"d2": ["cat renal failure stages"]})
    assert BM25().add(exp.keys(), exp.values()).search("renal")[0][0] == "d2"


def test_dense_and_rrf():
    d = DenseExact(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert d.search(np.array([0.9, 0.1]))[0][0] == "a"
    assert rrf([["a", "b", "c"], ["b", "c", "a"]])[0] == "b"


def test_held_out_slice_is_stable_and_sized():
    uids = [f"doc{i}" for i in range(20000)]
    s = held_out_slice(uids)
    assert 0.04 < len(s) / len(uids) < 0.06
    assert held_out_slice(uids[:10000]) == {u for u in s if u in set(uids[:10000])}


def test_oracle_is_upper_bound_on_beam(tmp_path):
    rng = np.random.default_rng(11)
    ids = sorted({SemanticId.from_ordinal(tuple(int(c) for c in rng.integers(0, 8, 4)), 0) for _ in range(800)})
    path, sha = write(ids, tmp_path, id_schema="ids_v1", corpus_snapshot="cs_2026_09_01")
    gold = {f"q{i}": ids[int(j)] for i, j in enumerate(rng.integers(0, len(ids), 40))}
    scorer = ToyScorer(gold, strength=1.0)
    with TrieSnapshot(path, sha) as trie:
        beam_hits = oracle_hits = 0
        for q, g in gold.items():
            beam = decode(scorer, q, trie, DecodeConfig(beam=4, prefixes_expanded=4, adaptive_beam=False))
            oracle = brute_force(scorer, q, trie)
            assert len(oracle.candidates) == len(ids)  # nothing pruned
            beam_hits += g in [c.sid for c in beam.candidates[:100]]
            oracle_hits += g in [c.sid for c in oracle.candidates[:100]]
        assert oracle_hits >= beam_hits
