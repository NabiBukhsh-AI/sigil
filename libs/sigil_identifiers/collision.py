"""Re-identification policy for content edits. §24 F4, §35.9.

A content edit can move a document across a quantization boundary. Re-identifying on
every such move churns identifiers for documents that sit near a boundary, and every
churn leaves an alias behind. Hysteresis: only re-identify when the new codes differ
*and* the new path reconstructs the embedding better than the old one by a margin. Past
``max_alias_depth`` re-identifications the document is pinned to a stable identifier,
accepting growing quantization error in exchange for not churning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigil_identifiers.quantizer.rq_kmeans import RQKMeans


@dataclass(frozen=True)
class Decision:
    reidentify: bool
    new_codes: tuple[int, ...]
    reason: str


def decide(
    q: RQKMeans,
    embedding: np.ndarray,
    old_codes: tuple[int, ...],
    *,
    alias_depth: int,
    margin: float = 0.05,
    max_alias_depth: int = 2,
) -> Decision:
    x = np.asarray(embedding, dtype=np.float32)[None]
    new = tuple(int(c) for c in q.encode(x)[0])
    if new == tuple(old_codes):
        return Decision(False, new, "codes_unchanged")
    if alias_depth >= max_alias_depth:
        return Decision(False, tuple(old_codes), "pinned_alias_depth")
    p = q.prep(x).astype(np.float64)
    d_old = float(((p - q.reconstruct(np.array([old_codes]))) ** 2).sum())
    d_new = float(((p - q.reconstruct(np.array([new]))) ** 2).sum())
    if d_old - d_new <= margin:
        return Decision(False, tuple(old_codes), "within_hysteresis_margin")
    return Decision(True, new, "moved_past_margin")
