"""Trie mask provider contract and the masking step. §4.4, §10.5.

The provider is anything with the ``TrieSnapshot`` decode-time surface: the numpy reader
in ``sigil_trie.mmap_reader`` or the Rust reader in ``sigil_trie/native``. Both must be
in-process (ADR 0003).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from sigil_core.errors import InvalidIdentifier


class MaskProvider(Protocol):
    ROOT: int

    def children_mask(self, nodes: np.ndarray, depth: int) -> np.ndarray: ...
    def child(self, nodes: np.ndarray, depth: int, codes: np.ndarray) -> np.ndarray: ...
    def leaves(self, nodes4: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def masked_log_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``z[k] <- z[k] if k in children else -inf``, then log-softmax over what remains.

    A row with no valid child is structurally impossible for a node that exists in the
    trie, so it is treated as F1 rather than silently producing NaNs.
    """
    if not mask.any(1).all():
        raise InvalidIdentifier("mask row with no valid children: beam holds a non-trie node")
    z = np.where(mask, logits.astype(np.float64), -np.inf)
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def assert_in_mask(mask: np.ndarray, rows: np.ndarray, codes: np.ndarray) -> None:
    """F1 guard. Runs in production, not only in tests: every emitted token must have been
    permitted by the trie. ``valid_id_rate`` is exactly 1.0 or this raises."""
    if not mask[rows, codes].all():
        raise InvalidIdentifier("decoder emitted a code the trie did not permit")
