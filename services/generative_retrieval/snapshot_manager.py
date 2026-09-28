"""Atomic trie snapshot swap. §15.6, §18.3, ADR 0003.

A swap maps and hash-verifies the new file first, then flips one pointer under a lock. In-
flight requests keep the snapshot they started with; the old one is closed when its last
reader releases it. A snapshot that fails verification is refused and the previous one
stays mapped (§24 F16).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

from sigil_core.errors import BundleIncompatible
from sigil_trie import TrieSnapshot


class _Ref:
    def __init__(self, trie: TrieSnapshot):
        self.trie, self.readers, self.retired = trie, 0, False


class SnapshotManager:
    def __init__(self, id_schema: str):
        self.id_schema = id_schema
        self._cur: _Ref | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._cur is not None

    @property
    def version(self) -> str | None:
        return self._cur.trie.version if self._cur else None

    @property
    def sha256(self) -> str | None:
        return self._cur.trie.sha256 if self._cur else None

    def swap(self, path: str | Path, sha256: str) -> str:
        new = TrieSnapshot(path, sha256)  # raises SnapshotIntegrityError before anything changes
        if new.id_schema != self.id_schema:
            new.close()
            raise BundleIncompatible(f"trie schema {new.id_schema} != serving schema {self.id_schema}")
        with self._lock:
            old, self._cur = self._cur, _Ref(new)
            if old is not None:
                old.retired = True
                if old.readers == 0:
                    old.trie.close()
        return new.version

    @contextmanager
    def acquire(self):
        with self._lock:
            ref = self._cur
            if ref is None:
                raise RuntimeError("no trie snapshot loaded")
            ref.readers += 1
        try:
            yield ref.trie
        finally:
            with self._lock:
                ref.readers -= 1
                if ref.retired and ref.readers == 0:
                    ref.trie.close()
