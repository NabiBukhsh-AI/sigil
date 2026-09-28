"""Duplicate handling. §3.1 I4, §7.4.

Exact duplicates collapse to one unit with multiple source pointers: two identical
documents with two identifiers force the model to split probability mass arbitrarily,
which is pure loss. Near duplicates (MinHash Jaccard 0.85 to 0.99) stay separate but
linked: the reranker deduplicates at serving time, and training treats linked units as
multi-positive rather than as hard negatives of each other.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from sigil_data.ingestion.clean import content_hash

_P = (1 << 31) - 1  # a, x < 2^31 so a*x + b fits in uint64 without overflow


def shingles(text: str, n: int = 5) -> np.ndarray:
    words = text.casefold().split()
    grams = {" ".join(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}
    return np.array(
        [int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") % _P for g in grams],
        dtype=np.uint64,
    )


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _P, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _P, num_perm, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        s = shingles(text)
        return ((self.a[:, None] * s[None, :] + self.b[:, None]) % _P).min(1)


def jaccard_estimate(s1: np.ndarray, s2: np.ndarray) -> float:
    return float((s1 == s2).mean())


def dedup(docs: Mapping[str, str], near_low: float = 0.85, near_high: float = 0.99,
          bands: int = 32, num_perm: int = 128) -> tuple[dict[str, list[str]], list[tuple[str, str, float]]]:
    """Returns ``(canonical -> [collapsed keys], near-duplicate links)``.

    Exact means identical canonical hash or estimated Jaccard >= ``near_high``.
    """
    by_hash: dict[str, list[str]] = defaultdict(list)
    for k in docs:
        by_hash[content_hash(docs[k])].append(k)
    canon = {keys[0]: keys for keys in by_hash.values()}

    mh = MinHasher(num_perm)
    sigs = {k: mh.signature(docs[k]) for k in canon}
    rows = num_perm // bands
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for k, s in sigs.items():
        for b in range(bands):
            buckets[(b, s[b * rows : (b + 1) * rows].tobytes())].append(k)
    pairs = {tuple(sorted((x, y))) for ks in buckets.values() for i, x in enumerate(ks) for y in ks[i + 1 :]}

    links, merged = [], {}
    for x, y in sorted(pairs):
        j = jaccard_estimate(sigs[x], sigs[y])
        if j >= near_high:
            merged[y] = merged.get(x, x)
        elif j >= near_low:
            links.append((x, y, j))
    out: dict[str, list[str]] = {}
    for k, keys in canon.items():
        out.setdefault(merged.get(k, k), []).extend(keys)
    links = [(merged.get(x, x), merged.get(y, y), j) for x, y, j in links if merged.get(x, x) != merged.get(y, y)]
    return out, links
