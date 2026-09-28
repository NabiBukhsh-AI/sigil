"""Dense baselines. §25.2. Offline evaluation only.

``DenseExact`` isolates the quantization-plus-generation loss from the embedding itself:
same embedding model as the quantizer, exhaustive inner product. ``DenseHNSW`` is the
production-realistic system this architecture replaces.

ADR 0006: these live in the eval library and never in anything under services/.
tests/architecture enforces that.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class DenseExact:
    def __init__(self, keys: Sequence[str], embeddings: np.ndarray):
        self.keys, self.emb = list(keys), _normalize(embeddings)

    def search(self, query_embedding: np.ndarray, n: int = 100) -> list[tuple[str, float]]:
        s = self.emb @ _normalize(query_embedding)
        top = np.lexsort((np.arange(len(s)), -s))[:n]
        return [(self.keys[i], float(s[i])) for i in top]


class DenseHNSW:
    def __init__(self, keys: Sequence[str], embeddings: np.ndarray, ef: int = 128, m: int = 32):
        import hnswlib  # optional, offline only

        emb = _normalize(embeddings)
        self.keys = list(keys)
        self.index = hnswlib.Index(space="ip", dim=emb.shape[1])
        self.index.init_index(max_elements=len(emb), ef_construction=200, M=m, random_seed=0)
        self.index.add_items(emb, np.arange(len(emb)))
        self.index.set_ef(ef)

    def search(self, query_embedding: np.ndarray, n: int = 100) -> list[tuple[str, float]]:
        labels, dists = self.index.knn_query(_normalize(query_embedding)[None], k=min(n, len(self.keys)))
        return [(self.keys[i], 1.0 - float(d)) for i, d in zip(labels[0], dists[0], strict=True)]
