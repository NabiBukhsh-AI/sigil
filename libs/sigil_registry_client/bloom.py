"""Tombstone Bloom filter. §12.3.

Pushed to every gateway pod within 2 seconds of a deletion, so a deleted document stops
being served before any cache expires. False positives cost a registry lookup, never a
wrong answer: the gateway always hard-checks the final k against the registry.
"""

from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Iterable

import numpy as np


class Bloom:
    def __init__(self, capacity: int = 100_000, error_rate: float = 1e-4, bits: np.ndarray | None = None,
                 k: int | None = None):
        m = max(64, int(-capacity * math.log(error_rate) / math.log(2) ** 2))
        self.k = k or max(1, round(m / capacity * math.log(2)))
        self.bits = bits if bits is not None else np.zeros((m + 7) // 8, dtype=np.uint8)
        self.m = len(self.bits) * 8

    def _idx(self, key: str) -> list[int]:
        h = hashlib.blake2b(key.encode(), digest_size=16).digest()
        a, b = int.from_bytes(h[:8], "little"), int.from_bytes(h[8:], "little") | 1
        return [(a + i * b) % self.m for i in range(self.k)]

    def add(self, key: str) -> None:
        for i in self._idx(key):
            self.bits[i >> 3] |= 1 << (i & 7)

    def update(self, keys: Iterable[str]) -> Bloom:
        for k in keys:
            self.add(k)
        return self

    def __contains__(self, key: str) -> bool:
        return all(self.bits[i >> 3] & (1 << (i & 7)) for i in self._idx(key))

    def to_wire(self) -> dict:
        return {"k": self.k, "bits": base64.b64encode(self.bits.tobytes()).decode()}

    @classmethod
    def from_wire(cls, d: dict) -> Bloom:
        bits = np.frombuffer(base64.b64decode(d["bits"]), dtype=np.uint8).copy()
        return cls(bits=bits, k=d["k"])
