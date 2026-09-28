"""Retrieval, generative-retrieval, and calibration metrics. §25.1.

Per-query functions take a ranked list of document keys and the query's graded
relevance judgments ``{doc: grade}`` (grade > 0 means relevant). ``evaluate_run``
averages them over a run, TREC style.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]


def recall_at(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    relevant = {d for d, g in rel.items() if g > 0}
    return len(relevant.intersection(ranked[:k])) / len(relevant) if relevant else 0.0


def precision_at(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    return sum(rel.get(d, 0) > 0 for d in ranked[:k]) / k


def mrr_at(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if rel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def success_at(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    """Query answered by at least one retrieved document. Correlates best with downstream
    LLM answer quality."""
    return float(any(rel.get(d, 0) > 0 for d in ranked[:k]))


def evaluate_run(run: Run, qrels: Qrels, *, queries: Iterable[str] | None = None) -> dict[str, float]:
    """The §25.1 retrieval-quality table, averaged over judged queries."""
    qids = [q for q in (queries or qrels) if q in qrels]
    fns = {
        "recall@10": (recall_at, 10), "recall@100": (recall_at, 100), "precision@5": (precision_at, 5),
        "mrr@10": (mrr_at, 10), "ndcg@10": (ndcg_at, 10), "hit@1": (success_at, 1),
        "success@10": (success_at, 10),
    }
    out = {name: float(np.mean([fn(list(run.get(q, [])), qrels[q], k) for q in qids])) if qids else 0.0
           for name, (fn, k) in fns.items()}
    out["n_queries"] = len(qids)
    return out


# -- generative-retrieval specific ----------------------------------------------------


def prefix_survival(gold_codes: Sequence[int], survivors: Sequence[np.ndarray]) -> int:
    """Deepest level at which the gold identifier still had a surviving prefix (0..4).

    ``prefix_survival@l`` is the share of queries with a value >= l: the single most
    diagnostic metric in the system, because it shows exactly where the beam loses
    documents that no downstream stage can recover.
    """
    depth = 0
    for level, prefixes in enumerate(survivors, start=1):
        if len(prefixes) and (prefixes == np.asarray(gold_codes[:level])).all(1).any():
            depth = level
        else:
            break
    return depth


def survival_curve(depths: Sequence[int], levels: int = 4) -> dict[str, float]:
    d = np.asarray(depths)
    return {f"prefix_survival@{l}": float((d >= l).mean()) if len(d) else 0.0 for l in range(1, levels + 1)}


def beam_recall_gap(beam_run: Run, oracle_run: Run, qrels: Qrels, k: int = 100) -> float:
    """Recall@k under brute-force scoring of every leaf minus Recall@k under the beam.
    A growing gap means the model is drifting from what the beam can reach."""
    return evaluate_run(oracle_run, qrels)[f"recall@{k}"] - evaluate_run(beam_run, qrels)[f"recall@{k}"]


def valid_id_rate(emitted: Iterable, trie) -> float:
    """Exactly 1.0, or it is a P1 bug."""
    emitted = list(emitted)
    return sum(s in trie for s in emitted) / len(emitted) if emitted else 1.0


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error, equal-width bins. Gate: under 0.05."""
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    idx = np.minimum((probs * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)


def kendall_tau(a: Sequence[str], b: Sequence[str]) -> float:
    """Rank agreement over the items both lists contain. §13.4 audits queries below 0.2."""
    pos = {x: i for i, x in enumerate(b)}
    common = [x for x in a if x in pos]
    n = len(common)
    if n < 2:
        return 1.0
    ranks = np.array([pos[x] for x in common])
    s = np.sign(ranks[None, :] - ranks[:, None])[np.triu_indices(n, 1)]
    return float(s.sum() / len(s))


def duplicate_rate(groups: Sequence) -> float:
    """``duplicate_rate@k`` given the near-duplicate group of each returned document."""
    return 1.0 - len(set(groups)) / len(groups) if groups else 0.0
