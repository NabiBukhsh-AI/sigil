"""Snapshot validation run by the trie builder before publishing and by tests.

Phase 3 round-trip property: every registry identifier is a decodable path in the trie,
and no other path exists. Combined with the per-step mask this is the whole of the §10.5
validity guarantee: for snapshot S, every decoded sequence is a root-to-leaf path in S.
"""

from __future__ import annotations

from collections.abc import Iterable

from sigil_core.ids import SemanticId
from sigil_trie.format import ids_digest
from sigil_trie.mmap_reader import TrieSnapshot


def validate(trie: TrieSnapshot, expected: Iterable[SemanticId]) -> list[str]:
    """Empty list means the snapshot is exactly the expected id set."""
    want = sorted(expected)
    problems = []
    if trie.header["ids_sha256"] != ids_digest([s.pack() for s in want]):
        problems.append("ids digest differs from registry snapshot")
    got = list(trie.iter_ids())
    if got != want:
        missing = set(want) - set(got)
        extra = set(got) - set(want)
        problems.append(f"{len(missing)} missing paths, {len(extra)} extra paths")
    if trie.n_leaves != len(want):
        problems.append(f"n_leaves {trie.n_leaves} != {len(want)}")
    return problems
