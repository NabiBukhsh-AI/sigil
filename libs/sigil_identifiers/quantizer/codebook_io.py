"""Versioned, hashed codebook serialization. ADR 0004: the codebooks are the only global
coupling in the design, so they are never loaded without verifying their hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from sigil_core.bundle import ArtefactMeta
from sigil_core.errors import SnapshotIntegrityError


def digest(codebooks: np.ndarray) -> str:
    arr = np.ascontiguousarray(codebooks, dtype="<f4")
    h = hashlib.sha256(json.dumps(list(arr.shape)).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def save(codebooks: np.ndarray, directory: str | Path, *, version: str, id_schema: str) -> ArtefactMeta:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(codebooks, dtype="<f4")
    np.save(d / "codebooks.npy", arr)
    meta = ArtefactMeta(version=version, sha256=digest(arr), id_schema=id_schema)
    (d / "meta.json").write_text(json.dumps(
        {"version": version, "sha256": meta.sha256, "id_schema": id_schema, "shape": list(arr.shape)},
        indent=2,
    ))
    return meta


def load(directory: str | Path, expected_sha256: str | None = None) -> tuple[np.ndarray, ArtefactMeta]:
    d = Path(directory)
    meta_raw = json.loads((d / "meta.json").read_text())
    arr = np.load(d / "codebooks.npy", allow_pickle=False)
    actual = digest(arr)
    for want, what in ((meta_raw["sha256"], "sidecar"), (expected_sha256, "manifest")):
        if want is not None and actual != want:
            raise SnapshotIntegrityError(f"codebook {meta_raw['version']}: {what} hash mismatch")
    return arr, ArtefactMeta(version=meta_raw["version"], sha256=actual, id_schema=meta_raw["id_schema"])
