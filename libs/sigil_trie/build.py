"""Compile a registry snapshot into an immutable trie file. §18.3, §20.2 S6.

The trie is a derived artefact of the registry and is never edited directly (§12.1).
Input is every non-tombstoned identifier in one id schema; output is a content-addressed
file whose name includes its hash, so publishing is a write of a new file, never an
overwrite of a mapped one.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from sigil_core.ids import LEVELS, SemanticId
from sigil_trie.format import FORMAT_VERSION, MAGIC, TAIL_WIDTH, align, array_names, ids_digest


def _rows(ids: Iterable[SemanticId]) -> np.ndarray:
    raw = [s.pack() for s in ids]
    width = LEVELS + TAIL_WIDTH
    buf = b"".join(r.ljust(width, b"\0") for r in raw)
    rows = np.frombuffer(buf, dtype=np.uint8).reshape(len(raw), width) if raw else np.zeros((0, width), np.uint8)
    rows = rows[np.lexsort(rows.T[::-1])]
    if len(rows) > 1 and (rows[1:] == rows[:-1]).all(1).any():
        raise ValueError("duplicate identifiers in snapshot; registry uniqueness violated")
    return rows


def compile_arrays(ids: Iterable[SemanticId]) -> dict[str, np.ndarray]:
    rows = _rows(ids)
    n = len(rows)
    # starts[d]: row index where each distinct length-d prefix begins. Depth 0 is the root,
    # which exists even in an empty trie.
    starts = [np.zeros(1, dtype=np.int64)]
    for d in range(1, LEVELS + 1):
        new = np.ones(n, dtype=bool)
        if n > 1:
            new[1:] = (rows[1:, :d] != rows[:-1, :d]).any(1)
        starts.append(np.flatnonzero(new))
    arrays: dict[str, np.ndarray] = {}
    for d in range(LEVELS):
        parent = np.searchsorted(starts[d], starts[d + 1], side="right") - 1
        arrays[f"child_off_{d}"] = np.searchsorted(parent, np.arange(len(starts[d]) + 1)).astype(np.uint32)
        arrays[f"child_code_{d}"] = rows[starts[d + 1], d].astype(np.uint8)
    arrays["leaf_off"] = np.append(starts[LEVELS], n).astype(np.uint32)
    arrays["leaf_tail"] = np.ascontiguousarray(rows[:, LEVELS:])
    return arrays


def write(
    ids: Iterable[SemanticId],
    directory: str | Path,
    *,
    id_schema: str,
    corpus_snapshot: str,
) -> tuple[Path, str]:
    """Write ``trie_<sha12>.trie`` plus a ``.sha256`` sidecar. Returns (path, sha256)."""
    ids = sorted(ids)
    arrays = compile_arrays(ids)
    layout, offset = {}, 0
    for name in array_names(LEVELS):
        a = arrays[name]
        layout[name] = {"dtype": a.dtype.str, "shape": list(a.shape), "offset": offset}
        offset = align(offset + a.nbytes)
    header = json.dumps({
        "format_version": FORMAT_VERSION,
        "id_schema": id_schema,
        "corpus_snapshot": corpus_snapshot,
        "levels": LEVELS,
        "n_leaves": int(len(arrays["leaf_tail"])),
        "ids_sha256": ids_digest([s.pack() for s in ids]),
        "arrays": layout,
    }, sort_keys=True).encode()
    preamble = MAGIC + struct.pack("<I", len(header)) + header
    body_start = align(len(preamble))

    h = hashlib.sha256()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".building-{os.getpid()}.trie"
    with open(tmp, "wb") as f:
        def put(b: bytes) -> None:
            f.write(b)
            h.update(b)

        put(preamble.ljust(body_start, b"\0"))
        pos = 0
        for name in array_names(LEVELS):
            spec = layout[name]
            put(b"\0" * (spec["offset"] - pos))
            data = arrays[name].tobytes()
            put(data)
            pos = spec["offset"] + len(data)
        f.flush()
        os.fsync(f.fileno())
    sha = h.hexdigest()
    final = directory / f"trie_{sha[:12]}.trie"
    if final.exists():  # identical content already published, possibly mapped right now
        tmp.unlink()
    else:
        os.replace(tmp, final)
    Path(str(final) + ".sha256").write_text(sha)
    return final, sha
