"""Phase 3 dataset checks: coverage floor, held-out exclusion, leakage, negative hygiene."""

from collections import Counter

import numpy as np
from sigil_core.config import load_yaml

from pipelines.dataset_build import build_rows
from scripts.local_stack import build, synthetic_corpus


def test_dataset_rows(tmp_path):
    docs = synthetic_corpus(8, 25)
    s = build(docs, tmp_path)
    text = {i: t for _, i, t in docs}
    corpus = []
    for i, parent in s.parent.items():
        r = s.reg.children(parent)[0]
        corpus.append({"doc_uid": r.doc_uid, "semantic_id": str(r.semantic_id), "title": r.title,
                       "text": text[i], "metadata": {"species": "canine"}})
    rng = np.random.default_rng(0)
    synthetic = {}
    for c, (t, i, _) in zip(corpus, docs, strict=True):
        # First candidate copies the source verbatim: must be rejected as leakage.
        qs = [" ".join(c["text"].split()[:12])]
        qs += [f"unique{i}{x} topic{t}word{j}" for x in "abc" for j in rng.choice(12, 4, replace=False)]
        synthetic[c["doc_uid"]] = [{"text": q, "query_type": "keyword", "generator": "g1"} for q in qs]
    cfg = load_yaml("configs/model/base.yaml")
    cfg["data"]["held_out"]["document_fraction"] = 0.1
    rows, rep = build_rows(corpus, synthetic, [], cfg, corpus_snapshot="cs_2026_09_01", id_schema="ids_v1")
    s.close()

    held = set(rep["held_out_doc_ids"])
    assert held and not held & {r["doc_id"] for r in rows}  # held-out docs contribute nothing
    per_unit = Counter(r["doc_id"] for r in rows)
    live = {c["doc_uid"] for c in corpus} - held
    assert min(per_unit[u] for u in live) >= 8 and rep["units_below_floor"] == 0
    assert rep["leaked"] >= len(live)  # every copied query was caught
    fam = rep["families"]
    assert fam["synthetic_query"] > fam["title"] and fam["content_prefix"] > 0 and fam["metadata"] > 0
    for r in rows:
        assert r["doc_id"] not in r["hard_negatives"]
        assert len(r["target_codes"]) == 5
        assert all(neg != r["target_codes"] for neg in r["hard_negative_codes"])
