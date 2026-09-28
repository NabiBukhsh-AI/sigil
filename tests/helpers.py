"""Shared test data builders."""

import zlib

import numpy as np


def clustered_embeddings(n: int, dim: int = 32, clusters: int = 40, seed: int = 0) -> np.ndarray:
    """Clumpy synthetic embeddings: real corpora are heavy-tailed, so cluster sizes are too."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim))
    sizes = rng.zipf(1.6, size=n) % clusters
    return (centers[sizes] + 0.35 * rng.normal(size=(n, dim))).astype(np.float32)


class ToyScorer:
    """Deterministic stand-in for the model: noisy logits with a bump on the gold path.
    Implements the ``sigil_decoding.Scorer`` protocol."""

    def __init__(self, gold: dict, strength: float = 3.0, noise: float = 1.0):
        self.gold, self.strength, self.noise = gold, strength, noise

    def encode(self, q):
        return q

    def step(self, q, prefixes):
        g = self.gold[q]
        level = prefixes.shape[1]
        out = np.empty((len(prefixes), 256))
        for i, p in enumerate(prefixes):
            seed = zlib.crc32(f"{q}|{bytes(p.astype(np.uint8)).hex()}".encode())
            out[i] = np.random.default_rng(seed).normal(size=256) * self.noise
        target = g.codes[level] if level < 4 else g.u
        out[(prefixes == np.array(g.codes[:level])).all(1), target] += self.strength
        return out
