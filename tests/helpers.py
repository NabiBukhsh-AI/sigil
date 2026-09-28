"""Shared test data builders."""

import zlib

import numpy as np


def clustered_embeddings(n: int, dim: int = 32, clusters: int = 40, seed: int = 0) -> np.ndarray:
    """Clumpy synthetic embeddings: real corpora are heavy-tailed, so cluster sizes are too."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim))
    sizes = rng.zipf(1.6, size=n) % clusters
    return (centers[sizes] + 0.35 * rng.normal(size=(n, dim))).astype(np.float32)


def hash_embed(texts, dim: int = 64) -> np.ndarray:
    """Deterministic signed bag-of-words hashing embedder. Test stand-in for the dense encoder."""
    import hashlib
    import re

    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
            out[i, h % dim] += 1.0 if (h >> 32) & 1 else -1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class RouterScorer:
    """The frozen quantizer used as the router: logits are negative distances from the
    query's residual to each level's centroids. A real, untrained generative router, which
    is exactly what a trained model improves on."""

    def __init__(self, quantizer, embed=hash_embed, temperature: float = 0.05):
        self.q, self.embed, self.t = quantizer, embed, temperature

    def encode(self, query):
        return self.q.prep(self.embed([query]))[0].astype(np.float64)

    def step(self, e, prefixes):
        n, level = prefixes.shape
        out = np.zeros((n, 256))
        if level == 4:
            return out  # u carries no content signal; fan-out and the reranker decide
        books = self.q.codebooks.astype(np.float64)
        R = np.repeat(e[None], n, 0)
        for l in range(level):
            R -= books[l][prefixes[:, l]]
        d = ((R[:, None, :] - books[level][None]) ** 2).sum(-1)
        out[:, : books.shape[1]] = -d / self.t
        return out


def overlap_reranker(query, texts):
    """Word-overlap cross-encoder stand-in, already on a [0, 1] scale."""
    import re

    q = set(re.findall(r"\w+", query.lower()))
    return np.array([len(q & set(re.findall(r"\w+", t.lower()))) / max(len(q), 1) for t in texts])


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
