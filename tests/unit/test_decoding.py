"""§33 decoding + property rows, Phase 5 acceptance: valid_id_rate == 1.0 by construction."""

import numpy as np
import pytest
from sigil_core.errors import InvalidIdentifier
from sigil_core.ids import ORDINALS_PER_PREFIX, SemanticId
from sigil_decoding import DecodeConfig, beam_search, decode
from sigil_decoding.confidence import Isotonic, fit_temperature, margin
from sigil_decoding.constraints import masked_log_softmax
from sigil_decoding.diversity import quota_select
from sigil_trie import TrieSnapshot, write

from tests.helpers import ToyScorer


class AdversarialScorer:
    """Puts all its mass on exactly the codes the trie forbids."""

    def __init__(self, trie):
        self.trie = trie

    def encode(self, q):
        return q

    def step(self, q, prefixes):
        out = np.empty((len(prefixes), 256))
        for i, p in enumerate(prefixes):
            depth = len(p)
            if depth == 4:
                out[i] = 50.0
                continue
            node = self.trie.find(tuple(int(c) for c in p))
            out[i] = np.where(self.trie.children_mask(np.array([node]), depth)[0], -50.0, 50.0)
        return out


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    rng = np.random.default_rng(3)
    ids = set()
    while len(ids) < 2500:
        codes = tuple(int(c) for c in rng.integers(0, 10, 4))
        ids.add(SemanticId.from_ordinal(codes, int(rng.integers(0, 4))))
    ids = sorted(ids)
    path, sha = write(ids, tmp_path_factory.mktemp("trie"), id_schema="ids_v1", corpus_snapshot="cs_2026_09_01")
    trie = TrieSnapshot(path, sha)
    gold = {f"q{i}": ids[int(j)] for i, j in enumerate(rng.integers(0, len(ids), 60))}
    yield ids, trie, gold
    trie.close()


def recall_at(result, g, k=100):
    return any(c.sid == g for c in result.candidates[:k])


def test_every_candidate_is_a_trie_path(corpus):
    ids, trie, gold = corpus
    scorer = ToyScorer(gold)
    for q in gold:
        r = decode(scorer, q, trie, DecodeConfig(beam=16))
        assert r.candidates and all(c.sid in trie for c in r.candidates)


def test_adversarial_scorer_cannot_escape_the_trie(corpus):
    ids, trie, gold = corpus
    r = decode(AdversarialScorer(trie), "q0", trie, DecodeConfig(beam=32))
    assert r.candidates and all(c.sid in trie for c in r.candidates)


def test_mask_rejects_non_trie_node():
    with pytest.raises(InvalidIdentifier):
        masked_log_softmax(np.zeros((1, 256)), np.zeros((1, 256), dtype=bool))


def test_beam_one_equals_greedy(corpus):
    ids, trie, gold = corpus
    scorer = ToyScorer(gold)
    cfg = DecodeConfig(beam=1, expansion=1, adaptive_beam=False)
    for q in list(gold)[:20]:
        out = beam_search(scorer, q, trie, cfg)
        node, prefix = np.array([trie.ROOT]), np.zeros((1, 0), dtype=np.int64)
        for depth in range(4):
            lp = masked_log_softmax(scorer.step(q, prefix), trie.children_mask(node, depth))
            c = int(np.argmax(lp[0]))
            node = trie.child(node, depth, np.array([c]))
            prefix = np.concatenate([prefix, [[c]]], 1)
        assert tuple(out.prefixes[0]) == tuple(prefix[0])


def test_larger_beam_never_decreases_recall(corpus):
    ids, trie, gold = corpus
    scorer = ToyScorer(gold, strength=1.2, noise=1.0)
    recalls = []
    for b in (2, 8, 32, 128):
        cfg = DecodeConfig(beam=b, adaptive_beam=False, prefixes_expanded=b, quota_l1=1.0,
                           candidate_cap=10**6, adaptive_candidate_cap=False)
        recalls.append(sum(recall_at(decode(scorer, q, trie, cfg), g, k=None) for q, g in gold.items()))
    assert recalls == sorted(recalls), recalls


def test_identical_requests_are_bitwise_identical(corpus):
    ids, trie, gold = corpus
    scorer = ToyScorer(gold)
    a = decode(scorer, "q5", trie, DecodeConfig())
    b = decode(scorer, "q5", trie, DecodeConfig())
    assert [(c.sid, c.score) for c in a.candidates] == [(c.sid, c.score) for c in b.candidates]
    assert np.array_equal(a.beam.prefixes, b.beam.prefixes)


def test_quota_caps_one_branch_and_backfills():
    ranked_groups = np.array([7, 7, 7, 7, 7, 1, 2])
    assert list(quota_select(ranked_groups, 5, 0.4)) == [0, 1, 2, 5, 6]
    assert list(quota_select(np.array([7] * 6), 5, 0.4)) == [0, 1, 2, 3, 4]  # backfill


def test_beam_respects_l1_quota_when_pool_has_room(corpus):
    ids, trie, gold = corpus
    cfg = DecodeConfig(beam=20, expansion=8, quota_l1=0.4, disable_quota_above_margin=1e9, adaptive_beam=False)
    out = beam_search(ToyScorer(gold, strength=6.0), "q1", trie, cfg)
    for survivors in out.survivors[1:]:
        _, counts = np.unique(survivors[:, 0], return_counts=True)
        assert counts.max() <= 8
    # With expansion=2 the peaked pool is all one branch: the quota backfills, it never
    # shrinks the beam.
    narrow = beam_search(ToyScorer(gold, strength=6.0), "q1", trie,
                         DecodeConfig(beam=20, expansion=2, quota_l1=0.4, disable_quota_above_margin=1e9,
                                      adaptive_beam=False))
    assert len(narrow.prefixes) == 20


def test_fanout_reaches_documents_whose_ordinal_was_never_learned(tmp_path):
    prefix = (4, 4, 4, 4)
    old = [SemanticId.from_ordinal(prefix, n) for n in range(3)]
    fresh = SemanticId.from_ordinal(prefix, 300)  # ESCAPE path, never seen by the "model"
    others = [SemanticId.from_ordinal((i, 0, 0, 0), 0) for i in range(10)]
    path, sha = write(old + [fresh] + others, tmp_path, id_schema="ids_v1", corpus_snapshot="cs_2026_09_01")
    with TrieSnapshot(path, sha) as trie:
        r = decode(ToyScorer({"q": old[0]}, strength=8.0), "q", trie, DecodeConfig(beam=4))
        assert fresh in [c.sid for c in r.candidates]


def test_adaptive_beam_widens_on_low_margin(corpus):
    ids, trie, gold = corpus
    flat = ToyScorer(gold, strength=0.0, noise=0.01)
    out = beam_search(flat, "q2", trie, DecodeConfig(beam=8, widened_beam=64, widen_below_margin=0.5))
    assert out.beam_width == 64


def test_temperature_fit_recovers_scale_and_isotonic_is_monotone():
    rng = np.random.default_rng(0)
    s = rng.normal(-3, 2, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(s / 2.0 + 1.0)))).astype(float)
    cal = fit_temperature(s, y)
    assert cal.temperature == pytest.approx(2.0, rel=0.15)
    iso = Isotonic.fit(s, y)
    grid = np.linspace(-10, 5, 50)
    assert (np.diff(iso(grid)) >= -1e-12).all()


def test_margin_of_single_child_is_capped():
    assert margin(np.array([0.0] + [-np.inf] * 255)) > 50
    assert ORDINALS_PER_PREFIX == 765
