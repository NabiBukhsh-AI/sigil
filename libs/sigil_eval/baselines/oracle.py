"""Brute-force generative scoring of every leaf. §10.7, §25.2, Experiment 11.

Rejected for serving, retained as an offline oracle: it measures the recall the beam is
losing, which is the single most informative diagnostic in the system. It is the same
decoder with the beam made wide enough that nothing is ever pruned, so the only
difference from serving is pruning.
"""

from __future__ import annotations

from typing import Any

from sigil_decoding import DecodeConfig, DecodeResult, Scorer, decode
from sigil_decoding.constraints import MaskProvider

UNBOUNDED = DecodeConfig(
    beam=10**9, expansion=1, prefixes_expanded=10**9, leaves_per_prefix=10**9,
    candidate_cap=10**9, adaptive_candidate_cap=False, quota_l1=1.0, adaptive_beam=False,
)


def brute_force(scorer: Scorer, query: Any, trie: MaskProvider, *, encoded: bool = False) -> DecodeResult:
    return decode(scorer, query, trie, UNBOUNDED, encoded=encoded)
