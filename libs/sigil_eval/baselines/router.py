"""Quantizer-as-router: the generative channel with zero learned parameters.

Logits at level ``l`` are negative distances from the query embedding's residual to that
level's centroids. It answers "how much does training add over routing by the frozen
quantizer alone?", and it gives lifecycle simulations, benchmarks, and CI a working
generative channel without a GPU or a checkpoint.

``HashEmbedder`` is a deterministic bag-of-words stand-in for the dense encoder, for the
same purposes. Neither is used in production serving.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_W = re.compile(r"\w+")


class HashEmbedder:
    def __init__(self, dim: int = 64):
        self.dim = dim

    def __call__(self, texts) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _W.findall(t.lower()):
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class RouterScorer:
    """Implements ``sigil_decoding.Scorer``."""

    def __init__(self, quantizer, embed=None, temperature: float = 0.05):
        self.q, self.embed, self.t = quantizer, embed or HashEmbedder(quantizer.codebooks.shape[-1]), temperature

    def encode(self, query: str) -> np.ndarray:
        return self.q.prep(self.embed([query]))[0].astype(np.float64)

    def step(self, e: np.ndarray, prefixes: np.ndarray) -> np.ndarray:
        n, level = prefixes.shape
        out = np.zeros((n, 256))
        if level == 4:
            return out  # the ordinal carries no content signal; fan-out and the reranker decide
        books = self.q.codebooks.astype(np.float64)
        R = np.repeat(e[None], n, 0)
        for l in range(level):
            R -= books[l][prefixes[:, l]]
        out[:, : books.shape[1]] = -((R[:, None, :] - books[level][None]) ** 2).sum(-1) / self.t
        return out


def overlap_reranker(query: str, texts) -> np.ndarray:
    """Word-overlap stand-in for the cross-encoder, on a [0, 1] scale."""
    q = set(_W.findall(query.lower()))
    return np.array([len(q & set(_W.findall(t.lower()))) / max(len(q), 1) for t in texts])
