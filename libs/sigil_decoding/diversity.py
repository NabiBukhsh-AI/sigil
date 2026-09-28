"""Prefix diversity quotas. §10.3, §11.2.

Diversity is enforced structurally rather than with a penalty term: the identifier
hierarchy already encodes semantic proximity, so "no more than 40 percent from one
level-1 code" is meaningful, cheap, and explainable.

A quota never shrinks the result below what is available. If the other branches cannot
fill the slots, the capped branch backfills them; the quota redistributes, it does not
discard.
"""

from __future__ import annotations

import numpy as np


def rank_within_group(groups: np.ndarray) -> np.ndarray:
    """For a ranked list, each item's rank among earlier items of the same group."""
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    rank = np.empty(len(groups), dtype=np.int64)
    rank[order] = np.arange(len(groups)) - np.searchsorted(g, g, side="left")
    return rank


def quota_select(groups: np.ndarray, n: int, quota: float) -> np.ndarray:
    """Pick ``n`` positions from an already-ranked list, at most ``quota * n`` per group.

    ``groups[i]`` is the group of the i-th ranked item. Returns positions into the ranked
    list, in rank order.
    """
    groups = np.asarray(groups)
    if quota >= 1.0 or len(groups) <= 1:
        return np.arange(min(n, len(groups)))
    ok = rank_within_group(groups) < max(1, int(quota * n))
    chosen = np.flatnonzero(ok)[:n]
    if len(chosen) < n:
        chosen = np.sort(np.concatenate([chosen, np.flatnonzero(~ok)[: n - len(chosen)]]))
    return chosen


def dedup(keys: list, scores: np.ndarray) -> np.ndarray:
    """Keep the best-scoring item per key (near-duplicate link group, simhash bucket).
    Returns kept positions in descending score order."""
    best: dict = {}
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        best.setdefault(keys[i], int(i))
    return np.array(sorted(best.values(), key=lambda i: (-scores[i], i)), dtype=np.int64)


def distinct_l1(codes_l1: list[int] | np.ndarray) -> int:
    """``distinct_level1_codes@k``: F11 prefix collapse detector."""
    return len(set(int(c) for c in codes_l1))
