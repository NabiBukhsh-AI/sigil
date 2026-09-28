"""Zero-copy trie loader with hash verification. ADR 0003: this lives in the decoder's
process, and every per-step call here is on the latency-critical path.

Beam state carries node ids, not prefixes, so a step needs no search: the children of
node ``i`` at depth ``d`` are one contiguous slice. All calls are vectorized over the beam.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import struct
from pathlib import Path

import numpy as np

from sigil_core.errors import SnapshotIntegrityError
from sigil_core.ids import ESCAPE, K, SemanticId
from sigil_trie.format import FORMAT_VERSION, MAGIC, TAIL_WIDTH, align


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _ranges(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated ``arange(s, e)`` for each pair, plus the owning row of each element."""
    lengths = (ends - starts).astype(np.int64)
    owner = np.repeat(np.arange(len(starts)), lengths)
    offsets = np.cumsum(lengths) - lengths
    flat = np.arange(lengths.sum()) - np.repeat(offsets, lengths) + np.repeat(starts.astype(np.int64), lengths)
    return flat, owner


class TrieSnapshot:
    ROOT = 0

    def __init__(self, path: str | Path, expected_sha256: str | None = None, verify: bool = True):
        self.path = Path(path)
        if verify:
            actual = file_sha256(self.path)
            sidecar = Path(str(self.path) + ".sha256")
            wants = [expected_sha256, sidecar.read_text().strip() if sidecar.exists() else None]
            if any(w is not None and w != actual for w in wants):
                raise SnapshotIntegrityError(f"{self.path.name}: sha256 mismatch, refusing to map")
            self.sha256 = actual
        else:
            self.sha256 = expected_sha256 or ""
        self._file = open(self.path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        if self._mm[:8] != MAGIC:
            self.close()
            raise SnapshotIntegrityError(f"{self.path.name}: bad magic")
        (hlen,) = struct.unpack("<I", self._mm[8:12])
        self.header = json.loads(self._mm[12 : 12 + hlen])
        if self.header["format_version"] != FORMAT_VERSION:
            self.close()
            raise SnapshotIntegrityError(f"unsupported trie format {self.header['format_version']}")
        body = align(12 + hlen)
        self.levels: int = self.header["levels"]
        self.id_schema: str = self.header["id_schema"]
        self.corpus_snapshot: str = self.header["corpus_snapshot"]
        self.version = f"trie_{self.sha256[:12]}"
        self._a = {
            name: np.frombuffer(self._mm, dtype=np.dtype(s["dtype"]), count=int(np.prod(s["shape"])),
                                offset=body + s["offset"]).reshape(s["shape"])
            for name, s in self.header["arrays"].items()
        }
        self.n_leaves: int = self.header["n_leaves"]
        self._keys: dict[int, np.ndarray] = {}

    # -- lifecycle ------------------------------------------------------------------------

    def close(self) -> None:
        self._a = {}
        self._keys = {}
        if getattr(self, "_mm", None) is not None:
            try:
                self._mm.close()
            except BufferError:
                pass  # a caller still holds a view; the OS unmaps when it is released
            self._mm = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> TrieSnapshot:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- decode-time API ------------------------------------------------------------------

    def children_mask(self, nodes: np.ndarray, depth: int) -> np.ndarray:
        """``bool[len(nodes), 256]``: which codes may follow each beam's prefix."""
        off = self._a[f"child_off_{depth}"]
        nodes = np.asarray(nodes, dtype=np.int64)
        flat, owner = _ranges(off[nodes], off[nodes + 1])
        mask = np.zeros((len(nodes), K), dtype=bool)
        mask[owner, self._a[f"child_code_{depth}"][flat]] = True
        return mask

    def _edge_keys(self, depth: int) -> np.ndarray:
        # Nodes at depth+1 are ordered by (parent, code), so parent*256+code is sorted and
        # one searchsorted resolves a whole beam. ponytail: 8 bytes per node of RAM (~80 MB
        # at 10M docs); the Rust reader does the per-slice search without it.
        if depth not in self._keys:
            off = self._a[f"child_off_{depth}"].astype(np.int64)
            parent = np.repeat(np.arange(len(off) - 1), np.diff(off))
            self._keys[depth] = parent * K + self._a[f"child_code_{depth}"]
        return self._keys[depth]

    def child(self, nodes: np.ndarray, depth: int, codes: np.ndarray) -> np.ndarray:
        """Node ids at ``depth + 1`` reached by taking ``codes`` from ``nodes``.
        Raises ``KeyError`` if any code is not a child, which a masked beam never does."""
        keys = self._edge_keys(depth)
        want = np.asarray(nodes, dtype=np.int64) * K + np.asarray(codes, dtype=np.int64)
        j = np.searchsorted(keys, want)
        ok = j < len(keys)
        ok[ok] = keys[j[ok]] == want[ok]
        if not ok.all():
            bad = int(np.flatnonzero(~ok)[0])
            raise KeyError(f"code {int(want[bad]) % K} is not a child of node {int(want[bad]) // K} at depth {depth}")
        return j

    def leaf_count(self, nodes4: np.ndarray) -> np.ndarray:
        off = self._a["leaf_off"]
        nodes4 = np.asarray(nodes4, dtype=np.int64)
        return (off[nodes4 + 1] - off[nodes4]).astype(np.int64)

    def leaves(self, nodes4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All tails under each depth-4 node: (``uint8[m, 3]`` tails, owner row per tail)."""
        off = self._a["leaf_off"]
        nodes4 = np.asarray(nodes4, dtype=np.int64)
        flat, owner = _ranges(off[nodes4], off[nodes4 + 1])
        return self._a["leaf_tail"][flat], owner

    # -- lookup and iteration (verification, tests, the explain tool) --------------------

    def find(self, codes: tuple[int, ...]) -> int | None:
        """Node id of a (possibly partial) prefix, or None if absent."""
        node = self.ROOT
        for depth, c in enumerate(codes):
            try:
                node = int(self.child(np.array([node]), depth, np.array([c]))[0])
            except KeyError:
                return None
        return node

    def __contains__(self, sid: SemanticId) -> bool:
        node = self.find(sid.codes)
        if node is None:
            return False
        tails, _ = self.leaves(np.array([node]))
        want = np.frombuffer(bytes(sid.tail).ljust(TAIL_WIDTH, b"\0"), dtype=np.uint8)
        return bool((tails == want).all(1).any())

    def iter_ids(self):
        """Every root-to-leaf path, in sorted order."""
        prefixes = [((), self.ROOT)]
        for depth in range(self.levels):
            off, cc = self._a[f"child_off_{depth}"], self._a[f"child_code_{depth}"]
            prefixes = [
                (p + (int(cc[j]),), j) for p, n in prefixes for j in range(int(off[n]), int(off[n + 1]))
            ]
        for p, n in prefixes:
            tails, _ = self.leaves(np.array([n]))
            for t in tails:
                yield SemanticId(p, tail_tuple(t))


def tail_tuple(t: np.ndarray) -> tuple[int, ...]:
    """Strip padding from a stored 3-byte tail using the ESCAPE chain."""
    out = []
    for b in t:
        out.append(int(b))
        if b != ESCAPE:
            break
    return tuple(out)
