"""Balanced residual quantizer: fit and apply. §5.1E, §5.3.

    c_l = argmin_k || r_{l-1} - C_l[k] ||,    r_l = r_{l-1} - C_l[c_l]

Each level refines the previous level's residual, so prefixes are coarse-to-fine by
construction and pruning a prefix at decode time prunes a semantically coherent region.

Fit uses balanced assignment at the coarse levels to shape the centroids. *Encoding*
is always plain nearest-centroid, so a document's codes are a pure function of its
embedding and the frozen codebooks: same document, same codes, in any process, at any
time (ADR 0004). Balance of the encoded corpus is measured by ``occupancy_report``,
not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sigil_identifiers.quantizer.balance import balanced_assign, nearest

KMEANSPP_SAMPLE = 50_000


def _kmeanspp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) > KMEANSPP_SAMPLE:
        X = X[rng.choice(len(X), KMEANSPP_SAMPLE, replace=False)]
    X = X.astype(np.float64)
    centers = [X[rng.integers(len(X))]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        total = d2.sum()
        i = rng.choice(len(X), p=d2 / total) if total > 0 else rng.integers(len(X))
        centers.append(X[i])
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    return np.stack(centers)


def _means(X: np.ndarray, assign: np.ndarray, k: int, C_prev: np.ndarray, dist: np.ndarray) -> np.ndarray:
    counts = np.bincount(assign, minlength=k)
    order = np.argsort(assign, kind="stable")
    starts = np.searchsorted(assign[order], np.arange(k))
    full = counts > 0
    C = C_prev.copy()
    C[full] = np.add.reduceat(X[order].astype(np.float64), starts[full]) / counts[full, None]
    # Empty clusters: reseed on the points worst served by their centroid. Deterministic.
    empty = np.flatnonzero(~full)
    if len(empty):
        worst = np.argsort(-dist, kind="stable")[: len(empty)]
        C[empty] = X[worst]
    return C


def _kmeans(X, k, rng, iters, balanced, max_ratio, penalty):
    C = _kmeanspp(X, k, rng)
    assign = None
    counts = None
    for _ in range(iters):
        if balanced:
            new = balanced_assign(X, C, max_ratio, penalty, counts)
        else:
            new, _ = nearest(X, C)
        dist = ((X - C[new]) ** 2).sum(1)
        if assign is not None and np.array_equal(new, assign):
            break
        assign = new
        counts = np.bincount(assign, minlength=k)
        C = _means(X, assign, k, C, dist)
    inertia = float(((X - C[assign]) ** 2).sum())
    return C, assign, inertia


@dataclass
class RQKMeans:
    levels: int = 4
    k: int = 256
    balanced_levels: tuple[int, ...] = (1, 2)
    max_ratio: float = 3.0
    penalty: float = 0.05
    iters: int = 50
    restarts: int = 3
    seed: int = 20260901
    normalize: bool = True
    codebooks: np.ndarray | None = field(default=None, repr=False)  # [L, K, D] float32

    @classmethod
    def from_config(cls, cfg: dict) -> RQKMeans:
        q, ids = cfg["quantizer"], cfg["identifiers"]
        return cls(
            levels=ids["levels"], k=ids["codes_per_level"], balanced_levels=tuple(q["balanced_levels"]),
            max_ratio=q["balance_max_ratio"], penalty=q["balance_penalty"], iters=q["kmeans_iters"],
            restarts=q["kmeans_restarts"], seed=q["kmeans_seed"], normalize=q["normalize_embeddings"],
        )

    def prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.normalize:
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X

    def fit(self, X: np.ndarray) -> RQKMeans:
        rng = np.random.default_rng(self.seed)
        R = self.prep(X).astype(np.float64)
        books = []
        for level in range(1, self.levels + 1):
            balanced = level in self.balanced_levels
            best = min(
                (_kmeans(R, self.k, rng, self.iters, balanced, self.max_ratio, self.penalty)
                 for _ in range(self.restarts)),
                key=lambda t: t[2],
            )
            C = best[0]
            books.append(C)
            codes, _ = nearest(R, C)
            R = R - C[codes]
        self.codebooks = np.stack(books).astype(np.float32)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """[n, D] embeddings -> [n, L] uint8 codes. Pure function of X and the codebooks."""
        if self.codebooks is None:
            raise RuntimeError("quantizer is not fit")
        R = self.prep(X).astype(np.float64)
        out = np.empty((len(R), self.levels), dtype=np.uint8)
        for level, C in enumerate(self.codebooks):
            c, _ = nearest(R, C)
            out[:, level] = c
            R = R - C.astype(np.float64)[c]
        return out

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.int64)
        return sum(self.codebooks[l].astype(np.float64)[codes[:, l]] for l in range(self.levels))

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared error. §24 F7: a 20% rise on new documents means corpus drift."""
        P = self.prep(X).astype(np.float64)
        return float(((P - self.reconstruct(self.encode(X))) ** 2).sum(1).mean())


def occupancy_report(codes: np.ndarray, k: int) -> dict:
    """Balance statistics for the Phase 2 acceptance: no level-1 code over 3x the mean."""
    codes = np.asarray(codes)
    l1 = np.bincount(codes[:, 0], minlength=k)
    l2 = np.unique(codes[:, :2], axis=0, return_counts=True)[1]
    leaves = np.unique(codes, axis=0, return_counts=True)[1]
    p = l1[l1 > 0] / l1.sum()
    return {
        "l1_max_over_mean": float(l1.max() / l1.mean()),
        "l1_used": int((l1 > 0).sum()),
        "l1_entropy_bits": float(-(p * np.log2(p)).sum()),
        "l2_max_over_mean": float(l2.max() / l2.mean()),
        "leaf_prefixes": int(len(leaves)),
        "leaf_max_occupancy": int(leaves.max()),
        "leaf_over_255": int((leaves > 255).sum()),
    }
