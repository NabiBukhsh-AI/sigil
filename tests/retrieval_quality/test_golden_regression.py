"""§33 retrieval quality regression: a frozen corpus with labelled queries runs in CI, and a
drop of more than 2 points of Recall@10 fails the build.

The corpus and generative channel are the deterministic synthetic stack, so the numbers
are stable across machines. When a change moves them on purpose, update baseline.json in
the same commit and say why.
"""

import json
from pathlib import Path

import pytest
from sigil_eval import metrics

from scripts.local_stack import build, query_for, synthetic_corpus

BASELINE = json.loads((Path(__file__).parent / "baseline.json").read_text())


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    docs = synthetic_corpus()
    s = build(docs, tmp_path_factory.mktemp("golden"))
    run, gen_run, qrels = {}, {}, {}
    for t, i, _ in docs[::2]:
        qid = f"q{i}"
        kids = {r.doc_uid for r in s.reg.children(s.parent[i])}
        qrels[qid] = dict.fromkeys(kids, 1)
        run[qid] = [r["doc_uid"] for r in s.ask(query_for(t, i))["results"]]
        gen = s.engine.retrieve(query_for(t, i), s.gw.cfg.decode)
        by_sid = {str(r.semantic_id): r.doc_uid for r in s.reg.records()}
        gen_run[qid] = [by_sid[c["semantic_id"]] for c in gen["candidates"] if c["semantic_id"] in by_sid]
    s.close()
    return run, gen_run, qrels


def test_end_to_end_recall_does_not_regress(run):
    got = metrics.evaluate_run(run[0], run[2])
    assert got["recall@10"] >= BASELINE["end_to_end_recall@10"] - 0.02, got
    assert got["ndcg@10"] >= BASELINE["end_to_end_ndcg@10"] - 0.02, got


def test_generative_channel_recall_does_not_regress(run):
    got = metrics.evaluate_run(run[1], run[2])
    assert got["recall@100"] >= BASELINE["generative_recall@100"] - 0.02, got
