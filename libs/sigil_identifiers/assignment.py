"""Ordinal allocation under a 4-level prefix: occupancy check, ``u``, and ESCAPE. §5.3.

Each prefix has a monotonic counter. The n-th document under a prefix gets ordinal n,
which maps to ``u = n`` for n < 255 and to ESCAPE chains after that. The counter never
goes down, so an identifier is never reused within a schema, even after deletion. That is
[FIXED] in §12.4: it prevents a stale model from resurrecting a deleted document as a
different one.

This is the in-process allocator used by tests, the dev registry, and offline bulk
assignment. services/registry implements the same contract as a Postgres row counter
(``INSERT .. ON CONFLICT DO UPDATE .. RETURNING``), whose row lock is the per-prefix
serialization that §24 F3 asks for.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

import numpy as np

from sigil_core.errors import PrefixCapacityExceeded
from sigil_core.ids import SemanticId


class PrefixCounter:
    def __init__(self) -> None:
        self._next: dict[tuple[int, ...], int] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_issued(cls, issued: Iterable[SemanticId]) -> PrefixCounter:
        """Rebuild from every identifier ever issued in the schema, tombstoned included."""
        c = cls()
        for sid in issued:
            c._next[sid.codes] = max(c._next.get(sid.codes, 0), sid.ordinal + 1)
        return c

    def allocate(self, codes: Iterable[int]) -> SemanticId:
        key = tuple(int(c) for c in codes)
        with self._lock:
            n = self._next.get(key, 0)
            try:
                sid = SemanticId.from_ordinal(key, n)
            except ValueError as e:
                raise PrefixCapacityExceeded(f"prefix {key} exhausted all ESCAPE levels") from e
            self._next[key] = n + 1
        return sid

    def allocate_batch(self, codes: np.ndarray) -> list[SemanticId]:
        return [self.allocate(row) for row in np.asarray(codes)]

    def occupancy(self, codes: Iterable[int]) -> int:
        return self._next.get(tuple(codes), 0)


def escape_rate(ids: Iterable[SemanticId]) -> float:
    """Share of identifiers that needed an ESCAPE level. Gate: under 0.5 percent."""
    ids = list(ids)
    return sum(s.escaped for s in ids) / len(ids) if ids else 0.0
