"""Phase 1 and 3 data checks: idempotent hashing, near-duplicate detection, chunk bounds,
query filters, coverage floor, sibling negatives, false-negative filtering, mixing."""

from collections import Counter

import numpy as np
import pytest
from sigil_core.ids import SemanticId
from sigil_data.ingestion.chunk import approx_tokens, chunk
from sigil_data.ingestion.clean import clean, content_hash, html_to_text, quality
from sigil_data.ingestion.dedup import dedup
from sigil_data.mixing import balance_units, coverage, mix
from sigil_data.negatives.mining import FNStats, PrefixIndex, assemble, drop_false_negatives
from sigil_data.query_generation.filters import LOW_COVERAGE, coverage_state, filter_queries, leaks
from sigil_eval.baselines.bm25 import BM25

TEXT = ("Clean soft tissue procedures under ninety minutes generally do not warrant post operative "
        "antibiotics in dogs. Prophylaxis should start within sixty minutes of incision and stop "
        "within twenty four hours. Longer courses increase resistance without improving outcomes.")


def test_html_and_hash_are_canonical():
    raw = f"<html><script>x()</script><p>{TEXT}</p></html>"
    assert "x()" not in html_to_text(raw)
    assert content_hash(TEXT) == content_hash("  " + TEXT.upper() + "\n")
    assert clean("a  b\x00c") == "a bc"
    assert quality("tiny")[0] is False and quality(TEXT)[0] is True


def test_dedup_collapses_exact_and_links_near():
    near = TEXT.replace("dogs", "canines")
    docs = {"a": TEXT, "b": TEXT + "  ", "c": near, "d": "Feline kidney disease staging follows IRIS guidelines " * 5}
    groups, links = dedup(docs, near_low=0.6)
    assert sorted(groups["a"]) == ["a", "b"]
    assert any({x, y} == {"a", "c"} for x, y, _ in links)
    assert not any("d" in (x, y) for x, y, _ in links)


def test_chunks_respect_the_512_token_bound():
    long = "\n\n".join([TEXT] * 40)
    units = chunk("doc1", long, max_tokens=120)
    assert len(units) > 1 and all(approx_tokens(u.text) <= 120 for u in units)
    assert units[1].key == "doc1#1"


def test_query_filters():
    docs = {"d1": TEXT, "d2": "Feline chronic kidney disease staging and diet.", "d3": "Equine hoof care basics."}
    bm25 = BM25().add(docs.keys(), docs.values())
    cands = [
        "how long should dogs get antibiotics after surgery",
        "how long should dogs get antibiotics after surgery?",  # duplicate
        "clean soft tissue procedures under ninety minutes generally do not warrant",  # leak
        "feline kidney staging",  # fails round trip for d1
        "prophylaxis timing incision",
    ]
    kept, st = filter_queries("d1", TEXT, cands, bm25)
    assert kept == [cands[0], cands[4]]
    assert (st.duplicate, st.leaked, st.round_trip_failed) == (1, 1, 1)
    assert leaks(cands[2], TEXT) and not leaks("dog antibiotics", TEXT)


def test_coverage_floor_escalates_then_routes_lexical():
    assert coverage_state(8, 0) == "ok"
    assert coverage_state(3, 0) == "regenerate"
    assert coverage_state(3, 2) == LOW_COVERAGE


def test_prefix_siblings_diverge_at_their_level():
    ids = {f"k{i}": SemanticId.from_ordinal(c, 0) for i, c in enumerate(
        [(1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 1), (1, 2, 1, 1), (2, 1, 1, 1)])}
    ids["k5"] = SemanticId.from_ordinal((1, 1, 1, 1), 1)
    idx = PrefixIndex(ids)
    sib = idx.siblings("k0", np.random.default_rng(0), per_level=5)
    assert set(sib) == {"k1", "k2", "k3", "k4", "k5"}


def test_false_negative_filter():
    st = FNStats()
    scores = {"pos": 0.9, "rel": 0.85, "irr": 0.1, "dup": 0.1}

    def ce(q, docs):
        return np.array([scores[d] for d in docs])

    kept = drop_false_negatives("q", "pos", ["rel", "irr", "dup"], ce, {"pos": {"dup"}}, stats=st)
    assert kept == ["irr"] and st.dropped_near_dup == 1 and st.dropped_score == 1
    assert drop_false_negatives("q", "pos", ["rel", "irr"], None, {}) == ["rel", "irr"]  # bootstrap round


def test_assemble_honours_shares():
    src = {"prefix_sibling": [f"s{i}" for i in range(50)], "bm25": [f"b{i}" for i in range(50)], "random": ["r0"]}
    out = assemble(src, {"prefix_sibling": 0.4, "bm25": 0.2, "random": 0.05}, 20, np.random.default_rng(0))
    assert len(out) == len(set(out)) == 20
    kinds = Counter(d[0] for d in out)
    assert kinds["s"] >= 8 and kinds["r"] == 1


def test_mixing_ratios_caps_and_upsampling():
    rows = [{"doc_id": f"d{i % 50}", "source": "synthetic_query", "weight": 1.0} for i in range(3000)]
    rows += [{"doc_id": f"d{i % 50}", "source": "title", "weight": 1.0} for i in range(500)]
    rows += [{"doc_id": "d0", "source": "real_query", "weight": 1.0}]
    bal = balance_units(rows, max_per_unit=20)
    assert max(coverage(bal).values()) <= 20
    d0 = [r for r in bal if r["doc_id"] == "d0"]
    d1 = [r for r in bal if r["doc_id"] == "d1"]
    assert any(r["weight"] == 3.0 for r in d0) and all(r["weight"] == 1.5 for r in d1)
    mixed = mix(rows, {"synthetic": 0.6, "titles": 0.1, "logs": 0.15}, 1000)
    fam = Counter(r["source"] for r in mixed)
    assert fam["real_query"] == 1  # logs short: remainder goes elsewhere, no repeats
    assert fam["synthetic_query"] >= 600


def test_dataset_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    from sigil_data import datasets

    rows = [{"input_text": f"query: q{i}", "target_codes": [1, 2, 3, 4, 0], "doc_id": f"d{i}",
             "source": "synthetic_query", "weight": 1.0, "corpus_snapshot": "cs_2026_09_01", "id_schema": "ids_v1"}
            for i in range(10)]
    m = datasets.write(rows, tmp_path, version="ds_test")
    assert m["rows"] == 10 and m["provenance"] == {"synthetic_query": 10}
    assert [r["doc_id"] for r in datasets.read(tmp_path / "ds_test")] == [f"d{i}" for i in range(10)]
    with pytest.raises(FileExistsError):
        datasets.write(rows, tmp_path, version="ds_test")  # versions are immutable
