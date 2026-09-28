"""Phase 3: trie round-trip property, mask correctness fuzz, integrity checks."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sigil_core.errors import SnapshotIntegrityError
from sigil_core.ids import ORDINALS_PER_PREFIX, SemanticId
from sigil_trie import TrieSnapshot, write
from sigil_trie.validate import validate


def random_ids(rng: np.random.Generator, n: int, spread: int = 6) -> list[SemanticId]:
    """Narrow code range so prefixes collide and ESCAPE paths occur."""
    out = set()
    while len(out) < n:
        codes = tuple(int(c) for c in rng.integers(0, spread, 4))
        out.add(SemanticId.from_ordinal(codes, int(rng.integers(0, ORDINALS_PER_PREFIX))))
    return sorted(out)


def build(tmp_path, ids):
    path, sha = write(ids, tmp_path, id_schema="ids_v1", corpus_snapshot="cs_2026_09_01")
    return TrieSnapshot(path, sha)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(0, 400))
def test_round_trip_every_path_and_nothing_else(tmp_path_factory, seed, n):
    ids = random_ids(np.random.default_rng(seed), n)
    with build(tmp_path_factory.mktemp("t"), ids) as trie:
        assert validate(trie, ids) == []
        for s in ids[:50]:
            assert s in trie


def test_masks_match_reference_fuzz(tmp_path):
    rng = np.random.default_rng(7)
    ids = random_ids(rng, 3000, spread=12)
    ref: dict[tuple, set] = {}
    for s in ids:
        for d in range(4):
            ref.setdefault(s.codes[:d], set()).add(s.codes[d])
    with build(tmp_path, ids) as trie:
        frontier = {(): TrieSnapshot.ROOT}
        for depth in range(4):
            prefixes = list(frontier)
            mask = trie.children_mask(np.array([frontier[p] for p in prefixes]), depth)
            nxt = {}
            for row, p in enumerate(prefixes):
                assert set(np.flatnonzero(mask[row])) == ref[p]
                assert mask[row].any()  # never an all -inf row
                codes = np.array(sorted(ref[p]))
                kids = trie.child(np.full(len(codes), frontier[p]), depth, codes)
                nxt.update({p + (int(c),): int(k) for c, k in zip(codes, kids)})
            frontier = nxt
        with pytest.raises(KeyError):
            trie.child(np.array([TrieSnapshot.ROOT]), 0, np.array([200]))


def test_leaves_enumerate_escape_tails(tmp_path):
    ids = [SemanticId.from_ordinal((1, 1, 1, 1), n) for n in (0, 254, 255, 700)]
    with build(tmp_path, ids) as trie:
        node = trie.find((1, 1, 1, 1))
        assert trie.leaf_count(np.array([node]))[0] == 4
        assert sorted(trie.iter_ids()) == ids


def test_corrupt_byte_is_refused(tmp_path):
    ids = random_ids(np.random.default_rng(1), 50)
    path, sha = write(ids, tmp_path, id_schema="ids_v1", corpus_snapshot="cs_2026_09_01")
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    bad = tmp_path / "bad.trie"
    bad.write_bytes(bytes(raw))
    with pytest.raises(SnapshotIntegrityError):
        TrieSnapshot(bad, sha)


def test_content_addressed_and_idempotent(tmp_path):
    ids = random_ids(np.random.default_rng(2), 20)
    p1, s1 = write(ids, tmp_path, id_schema="ids_v1", corpus_snapshot="cs_2026_09_01")
    with TrieSnapshot(p1, s1):
        p2, s2 = write(reversed(ids), tmp_path, id_schema="ids_v1", corpus_snapshot="cs_2026_09_01")
    assert (p1, s1) == (p2, s2)


def test_duplicates_rejected(tmp_path):
    s = SemanticId.parse("1.2.3.4.0")
    with pytest.raises(ValueError):
        write([s, s], tmp_path, id_schema="ids_v1", corpus_snapshot="cs_2026_09_01")
