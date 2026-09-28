"""Negative mining and false-negative filtering. §9.

Shares (§9.1): prefix siblings 40%, self-negatives 25%, BM25 20%, dense 10%, random 5%.
Self-negatives come from ``sigil_training.stages.mine_self_negatives`` against the live
checkpoint; everything else is mined here, offline.

Prefix siblings are the highest-value class: documents sharing ``c_<i`` with the positive
and diverging at level ``i`` are exactly the competitors the beam must beat at step ``i``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sigil_core.ids import LEVELS, SemanticId


class PrefixIndex:
    """doc keys grouped by every prefix length, for sibling lookup."""

    def __init__(self, ids: Mapping[str, SemanticId]):
        self.ids = dict(ids)
        self._by: dict[tuple, list[str]] = defaultdict(list)
        for key, sid in sorted(self.ids.items()):
            full = (*sid.codes, sid.u)
            for n in range(LEVELS + 1):
                self._by[full[:n]].append(key)

    def siblings(self, key: str, rng: np.random.Generator, per_level: int = 2) -> list[str]:
        """For each level ``i``, up to ``per_level`` documents that share the first ``i-1``
        codes and differ at ``i``."""
        sid = self.ids[key]
        full = (*sid.codes, sid.u)
        out: list[str] = []
        for i in range(LEVELS + 1):
            pool = [k for k in self._by[full[:i]] if k != key and self._code(k, i) != full[i]]
            if pool:
                out += list(rng.choice(pool, size=min(per_level, len(pool)), replace=False))
        return out

    def _code(self, key: str, i: int) -> int:
        s = self.ids[key]
        return (*s.codes, s.u)[i]


def lexical(query: str, bm25, positives: set[str], n: int = 100) -> list[str]:
    return [k for k, _ in bm25.search(query, n + len(positives)) if k not in positives][:n]


def dense(query_embedding: np.ndarray, index, positives: set[str], n: int = 100) -> list[str]:
    """ADR 0006: ``index`` is a transient ``sigil_eval.baselines.dense`` index built inside
    the training job and discarded with it. Never deployed."""
    return [k for k, _ in index.search(query_embedding, n + len(positives)) if k not in positives][:n]


@dataclass
class FNStats:
    seen: int = 0
    dropped_score: int = 0
    dropped_near_dup: int = 0

    @property
    def discard_rate(self) -> float:
        return (self.dropped_score + self.dropped_near_dup) / self.seen if self.seen else 0.0


def drop_false_negatives(
    query: str,
    positive: str,
    negatives: Sequence[str],
    score: Callable[[str, list[str]], np.ndarray] | None,
    near_dups: Mapping[str, set[str]],
    tau_fn: float = 0.7,
    delta: float = 0.1,
    stats: FNStats | None = None,
) -> list[str]:
    """§9.3. Mined negatives are frequently relevant, and training on them teaches the
    model to suppress correct answers. ``score(query, docs)`` is the calibrated
    cross-encoder; ``None`` during the first bootstrap round, when no reranker exists and
    only near-duplicate links are applied."""
    stats = stats or FNStats()
    linked = near_dups.get(positive, set())
    cand = [d for d in negatives if d not in linked]
    stats.seen += len(negatives)
    stats.dropped_near_dup += len(negatives) - len(cand)
    if score is None or not cand:
        return cand
    s = score(query, [positive, *cand])
    keep = [d for d, v in zip(cand, s[1:], strict=True) if v <= tau_fn and v <= s[0] - delta]
    stats.dropped_score += len(cand) - len(keep)
    return keep


def assemble(sources: Mapping[str, Iterable[str]], shares: Mapping[str, float], total: int,
             rng: np.random.Generator) -> list[str]:
    """Draw ``total`` distinct negatives by source share; shortfalls are refilled from
    whichever sources still have candidates, in share order."""
    pools = {s: list(dict.fromkeys(v)) for s, v in sources.items()}
    out: list[str] = []
    seen: set[str] = set()
    for src, share in sorted(shares.items(), key=lambda t: -t[1]):
        want = int(round(share * total))
        for d in pools.get(src, []):
            if want == 0:
                break
            if d not in seen:
                out.append(d)
                seen.add(d)
                want -= 1
    leftovers = [d for src, _ in sorted(shares.items(), key=lambda t: -t[1]) for d in pools.get(src, []) if d not in seen]
    out += list(dict.fromkeys(leftovers))[: max(0, total - len(out))]
    rng.shuffle(out)
    return out[:total]
