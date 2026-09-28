"""Build a versioned training dataset. §3.1 T1-T9, §7, §9.

    python pipelines/dataset_build.py --corpus export/corpus.jsonl --synthetic export/synthetic.jsonl \
        --logs export/logs.jsonl --logs-before 2026-08-01 --version ds_v1 --out datasets

corpus.jsonl rows:    {"doc_uid", "semantic_id", "title", "text", "metadata", "near_dup_group"?}
synthetic.jsonl rows: {"doc_uid", "queries": [{"text", "query_type", "generator", "prompt"}]}
logs.jsonl rows:      {"query", "doc_uid", "ts"}

Synthetic queries are generated as a separate batch job (``--generator`` runs it inline for
small corpora). Held-out documents (§7.5) contribute no examples at all, so recall on them
measures what a never-trained document gets: the cold-start capability itself.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from sigil_core.config import load_yaml
from sigil_core.ids import SemanticId
from sigil_data import datasets
from sigil_data.mixing import balance_units, coverage, family_shares, mix
from sigil_data.negatives.mining import FNStats, PrefixIndex, assemble, drop_false_negatives, lexical
from sigil_data.query_generation.filters import LOW_COVERAGE, coverage_state, filter_queries
from sigil_eval.baselines.bm25 import BM25
from sigil_eval.golden_sets import held_out_slice
from sigil_model.tokenization import format_document, format_query

PREFIX_WORDS = 48  # ~64 tokens of content prefix (§7.2)


def _jsonl(path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()] if path else []


def build_rows(corpus: list[dict], synthetic: dict[str, list[dict]], logs: list[dict], cfg: dict, *,
               corpus_snapshot: str, id_schema: str, logs_before: str | None = None, seed: int = 0) -> tuple[list[dict], dict]:
    d = cfg["data"]
    sq = d["synthetic_query_generation"]
    rng = np.random.default_rng(seed)
    docs = {r["doc_uid"]: r for r in corpus}
    sids = {u: SemanticId.parse(r["semantic_id"]) for u, r in docs.items()}
    held = held_out_slice(docs, d["held_out"]["document_fraction"], d["held_out"]["seed"])
    bm25 = BM25().add(docs.keys(), (f"{r.get('title') or ''}\n{r['text']}" for r in docs.values()))
    groups: dict[str, set[str]] = {}
    for u, r in docs.items():
        if g := r.get("near_dup_group"):
            groups.setdefault(g, set()).add(u)
    near = {u: groups.get(docs[u].get("near_dup_group"), set()) - {u} for u in docs}
    prefix_index = PrefixIndex(sids)

    def row(u: str, text: str, source: str, **extra) -> dict:
        s = sids[u]
        return {"input_text": text, "target_codes": [*s.codes, s.u], "doc_id": u, "source": source,
                "weight": 1.0, "corpus_snapshot": corpus_snapshot, "id_schema": id_schema,
                "hard_negatives": [], "hard_negative_codes": [], **extra}

    rows, stats, low = [], Counter(), []
    for u, r in docs.items():
        if u in held:
            stats["held_out_docs"] += 1
            continue
        cands = [q["text"] for q in synthetic.get(u, [])]
        meta = {q["text"]: q for q in synthetic.get(u, [])}
        kept, st = filter_queries(u, r["text"], cands, bm25, keep=d["queries_per_doc"],
                                  dedup_jaccard=sq["dedup_trigram_jaccard"], roundtrip_top_n=sq["roundtrip_filter_top_n"],
                                  ngram_n=sq["max_ngram_overlap_n"], ngram_ratio=sq["max_ngram_overlap_ratio"])
        stats.update(generated=st.generated, duplicate=st.duplicate, leaked=st.leaked,
                     round_trip_failed=st.round_trip_failed, accepted=st.accepted)
        if coverage_state(len(kept), attempts=2, min_per_unit=d["min_queries_per_doc"]) == LOW_COVERAGE and cands:
            low.append(u)
        for q in kept:
            m = meta.get(q, {})
            rows.append(row(u, format_query(q), "synthetic_query", generator=m.get("generator"),
                            query_type=m.get("query_type")))
        if r.get("title"):
            rows.append(row(u, format_query(r["title"]), "title"))
        rows.append(row(u, format_document(" ".join(r["text"].split()[:PREFIX_WORDS])), "content_prefix"))
        if r.get("metadata"):
            rendered = "; ".join(f"{k}: {v}" for k, v in sorted(r["metadata"].items()) if k != "chunk")
            if rendered:
                rows.append(row(u, format_query(rendered), "metadata"))

    for lg in logs:  # §7.5: temporal split; train only on queries before the cut
        if lg["doc_uid"] in docs and lg["doc_uid"] not in held and (logs_before is None or lg["ts"] < logs_before):
            rows.append(row(lg["doc_uid"], format_query(lg["query"]), "real_query"))

    # Negatives (§9.1). Self-negatives come later from stage C; dense from an offline index
    # when one is supplied. Shortfalls refill from the other sources.
    fn = FNStats()
    shares = d["negatives"]
    for r in rows:
        if r["source"] == "content_prefix":
            continue
        u = r["doc_id"]
        q = r["input_text"].split("query: ", 1)[-1]
        pos = {u} | near[u]
        sib = [x for x in prefix_index.siblings(u, rng) if x not in pos]
        lex = drop_false_negatives(q, u, lexical(q, bm25, pos, 20), None, near, stats=fn)
        rnd = [x for x in rng.choice(list(docs), min(8, len(docs)), replace=False) if x not in pos]
        negs = assemble({"prefix_sibling": sib, "bm25": lex, "random": rnd}, shares, 8, rng)
        r["hard_negatives"] = negs
        r["hard_negative_codes"] = [[*sids[x].codes, sids[x].u] for x in negs]

    rows = balance_units(rows, d["max_queries_per_doc"] + 3, d["zero_log_coverage_upsample"], seed)
    rows = mix(rows, d["family_ratios"])
    cov = coverage(rows)
    report = {**stats, "rows": len(rows), "low_coverage_docs": low, "fn_discard_rate": round(fn.discard_rate, 4),
              "families": dict(Counter(r["source"] for r in rows)),
              "family_weight_shares": {k: round(v, 4) for k, v in family_shares(rows).items()},
              "units_below_floor": sum(1 for u in docs if u not in held and cov.get(u, 0) < d["min_queries_per_doc"]),
              "held_out_doc_ids": sorted(held)}
    return rows, report


def generate_inline(corpus: list[dict], model: str, cfg: dict) -> dict[str, list[dict]]:
    from sigil_data.query_generation.generator import Doc2Query, interleave_types

    sq = cfg["data"]["synthetic_query_generation"]
    gen = Doc2Query(model, instruction="instruct" in model.lower())
    out = {}
    for i, r in enumerate(corpus):
        family = "a" if i % 2 == 0 else "b"  # two prompt families at minimum (§7.5)
        qs = gen.generate(r["text"], n=sq["n_generated"], temperature=sq["temperature"], top_p=sq["top_p"],
                          prompt_family=family, seed=i)
        out[r["doc_uid"]] = [q.__dict__ for q in interleave_types(qs)]
    return out


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/model/base.yaml")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--synthetic")
    ap.add_argument("--generator", help="run doc2query inline with this HF model")
    ap.add_argument("--logs")
    ap.add_argument("--logs-before")
    ap.add_argument("--corpus-snapshot", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", default="datasets")
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    corpus = _jsonl(a.corpus)
    synthetic = {r["doc_uid"]: r["queries"] for r in _jsonl(a.synthetic)}
    if a.generator:
        synthetic = generate_inline(corpus, a.generator, cfg)
    rows, report = build_rows(corpus, synthetic, _jsonl(a.logs), cfg, corpus_snapshot=a.corpus_snapshot,
                              id_schema=cfg["id_schema"], logs_before=a.logs_before)
    manifest = datasets.write(rows, a.out, version=a.version)
    (Path(a.out) / a.version / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({**manifest, "low_coverage": len(report["low_coverage_docs"]),
                      "units_below_floor": report["units_below_floor"]}, indent=2))


if __name__ == "__main__":
    main()
