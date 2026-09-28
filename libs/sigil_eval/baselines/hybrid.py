"""The decision baseline. §25.2: BM25 + dense + RRF + the same cross-encoder.

SIGIL must land within 3 points of NDCG@10 of this to justify itself, and must win
clearly on index footprint or on downstream LLM fusion to be preferred.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def rrf(runs: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion. Ties break by first appearance, so fusion is deterministic."""
    score: dict[str, float] = {}
    first: dict[str, int] = {}
    for run in runs:
        for rank, doc in enumerate(run):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank + 1)
            first.setdefault(doc, len(first))
    return sorted(score, key=lambda d: (-score[d], first[d]))


def hybrid(
    lexical: Sequence[str],
    dense: Sequence[str],
    rerank: Callable[[list[str]], list[float]] | None = None,
    depth: int = 100,
) -> list[str]:
    fused = rrf([lexical, dense])[:depth]
    if rerank is None:
        return fused
    scores = rerank(fused)
    return [d for _, d in sorted(zip(scores, fused, strict=True), key=lambda t: -t[0])]
