"""Filters on generated queries. §7.3, §7.4, §7.5.

Order matters for cost: cheap string filters first, the BM25 round trip last.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sigil_eval.baselines.bm25 import BM25, tokenize

LOW_COVERAGE = "LOW_COVERAGE"


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def trigram_jaccard(a: str, b: str) -> float:
    ta, tb = _ngrams(tokenize(a), 3), _ngrams(tokenize(b), 3)
    if not ta or not tb:
        return float(tokenize(a) == tokenize(b))
    return len(ta & tb) / len(ta | tb)


def leaks(query: str, doc: str, n: int = 6, max_ratio: float = 0.5) -> bool:
    """§7.5 item 4: a query that copies a long span of its source teaches copying, not
    retrieval. Short queries (under n tokens) cannot contain an n-gram and pass."""
    q = _ngrams(tokenize(query), n)
    return bool(q) and len(q & _ngrams(tokenize(doc), n)) / len(q) > max_ratio


@dataclass
class FilterStats:
    generated: int = 0
    duplicate: int = 0
    leaked: int = 0
    round_trip_failed: int = 0
    accepted: int = 0
    reasons: dict = field(default_factory=dict)


def filter_queries(
    doc_key: str,
    doc_text: str,
    candidates: Sequence[str],
    bm25: BM25,
    *,
    keep: int = 10,
    dedup_jaccard: float = 0.8,
    roundtrip_top_n: int = 50,
    ngram_n: int = 6,
    ngram_ratio: float = 0.5,
) -> tuple[list[str], FilterStats]:
    """Dedup, leakage, then round-trip consistency: keep ``q'`` only if BM25 over the
    corpus ranks the source in its top ``roundtrip_top_n``. That removes queries describing
    content the document does not contain; expect to discard 15 to 30 percent."""
    st = FilterStats(generated=len(candidates))
    kept: list[str] = []
    for q in candidates:
        if len(kept) >= keep:
            break
        if any(trigram_jaccard(q, k) > dedup_jaccard for k in kept):
            st.duplicate += 1
        elif leaks(q, doc_text, ngram_n, ngram_ratio):
            st.leaked += 1
        elif doc_key not in {k for k, _ in bm25.search(q, roundtrip_top_n)}:
            st.round_trip_failed += 1
        else:
            kept.append(q)
    st.accepted = len(kept)
    return kept, st


def coverage_state(accepted: int, attempts: int, min_per_unit: int = 8, max_attempts: int = 2) -> str:
    """[FIXED] §7.4 coverage floor. ``ok``; ``regenerate`` with a stronger generator; or
    ``LOW_COVERAGE``, which permanently routes the unit to the lexical channel. Some
    documents cannot be described by a generated query, and pretending otherwise silently
    destroys recall."""
    if accepted >= min_per_unit:
        return "ok"
    return "regenerate" if attempts < max_attempts else LOW_COVERAGE
