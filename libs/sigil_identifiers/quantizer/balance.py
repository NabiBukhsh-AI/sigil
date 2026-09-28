"""Capacity-constrained assignment for the balanced levels of the quantizer.

§5.3: levels 1 and 2 are fit with balanced k-means so no coarse partition exceeds
``max_ratio`` times the mean occupancy. A soft capacity penalty from the previous
iteration's counts pulls centroids toward balance; a hard cap guarantees it.
"""

from __future__ import annotations

import numpy as np

CHUNK = 32768


def sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared L2 distances in float64. float64 keeps argmin stable across platforms."""
    X = X.astype(np.float64, copy=False)
    C = C.astype(np.float64, copy=False)
    d = (X * X).sum(1)[:, None] - 2.0 * (X @ C.T) + (C * C).sum(1)[None, :]
    return np.maximum(d, 0.0)


def nearest(X: np.ndarray, C: np.ndarray, bias: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Index of and (biased) distance to the nearest centroid, computed in chunks."""
    idx = np.empty(len(X), dtype=np.int64)
    dist = np.empty(len(X), dtype=np.float64)
    for s in range(0, len(X), CHUNK):
        d = sq_dists(X[s : s + CHUNK], C)
        if bias is not None:
            d += bias
        i = d.argmin(1)
        idx[s : s + CHUNK] = i
        dist[s : s + CHUNK] = d[np.arange(len(d)), i]
    return idx, dist


def capacity(n: int, k: int, max_ratio: float) -> int:
    return max(int(max_ratio * n / k), -(-n // k))  # never infeasible


def balanced_assign(
    X: np.ndarray,
    C: np.ndarray,
    max_ratio: float,
    penalty: float = 0.0,
    prev_counts: np.ndarray | None = None,
) -> np.ndarray:
    """Assign every row of X to a centroid with no centroid over capacity.

    Each round, pending points pick their nearest centroid that still has room; each
    centroid accepts its closest applicants up to its remaining room and the rest retry.
    Every round accepts at least one point, so it terminates. Deterministic: ties break
    by centroid index, then by distance, then by row index.
    """
    n, k = len(X), len(C)
    room = np.full(k, capacity(n, k, max_ratio), dtype=np.int64)
    bias = np.zeros(k)
    if penalty and prev_counts is not None:
        scale = float(np.mean((X.astype(np.float64) ** 2).sum(1))) or 1.0
        bias = penalty * scale * prev_counts / max(n / k, 1.0)

    assign = np.full(n, -1, dtype=np.int64)
    pending = np.arange(n)
    while len(pending):
        choice, d = nearest(X[pending], C, bias + np.where(room > 0, 0.0, np.inf))
        order = np.lexsort((pending, d, choice))
        ch = choice[order]
        rank = np.arange(len(ch)) - np.searchsorted(ch, ch, side="left")
        ok = rank < room[ch]
        assign[pending[order[ok]]] = ch[ok]
        room -= np.bincount(ch[ok], minlength=k)
        pending = np.sort(pending[order[~ok]])
    return assign
