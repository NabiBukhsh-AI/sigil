"""§17.3 compatibility rules and manifest signing."""

import pytest
from sigil_core.bundle import ArtefactMeta, BundleManifest, check_compatibility
from sigil_core.errors import BundleIncompatible


def manifest(**over):
    base = dict(
        bundle_id="bundle_2026_09_02_a", backbone="bb_v2", adapter="ad_v37", id_schema="ids_v1",
        codebooks="cb_v1.0", codebook_sha256="c0de", trie_snapshot="trie_v1", trie_sha256="7e1e",
        corpus_snapshot="cs_2026_08_31", corpus_epoch_range=(1, 9), reranker="rr_v1", dataset="ds_v1",
    )
    return BundleManifest(**{**base, **over})


CB = ArtefactMeta("cb_v1.0", "c0de", id_schema="ids_v1")
TRIE = ArtefactMeta("trie_v1", "7e1e", id_schema="ids_v1", corpus_snapshot="cs_2026_08_31")
AD = ArtefactMeta("ad_v37", backbone="bb_v2", trained_on_snapshot="cs_2026_08_29")


def test_compatible_bundle_passes():
    assert check_compatibility(manifest(), codebook=CB, trie=TRIE, adapter=AD) == []


@pytest.mark.parametrize(
    "kw",
    [
        dict(codebook=ArtefactMeta("cb_v1.0", "bad", id_schema="ids_v1"), trie=TRIE, adapter=AD),
        dict(codebook=CB, trie=ArtefactMeta("trie_v1", "7e1e", id_schema="ids_v2"), adapter=AD),
        dict(codebook=CB, trie=TRIE, adapter=ArtefactMeta("ad_v37", backbone="bb_v1")),
        dict(codebook=CB, trie=TRIE, adapter=None),
    ],
)
def test_hard_violations_refuse(kw):
    with pytest.raises(BundleIncompatible):
        check_compatibility(manifest(), **kw)


def test_snapshot_gap_warns_then_alarms():
    old = ArtefactMeta("ad_v37", backbone="bb_v2", trained_on_snapshot="cs_2026_08_20")
    assert check_compatibility(manifest(), codebook=CB, trie=TRIE, adapter=old)[0].startswith("WARN")
    older = ArtefactMeta("ad_v37", backbone="bb_v2", trained_on_snapshot="cs_2026_07_01")
    assert check_compatibility(manifest(), codebook=CB, trie=TRIE, adapter=older)[0].startswith("ALARM")


def test_signature_round_trip_and_tamper(tmp_path):
    m = manifest().signed(b"k")
    assert m.verify_signature(b"k")
    m.save(tmp_path / "m.json")
    loaded = BundleManifest.load(tmp_path / "m.json")
    assert loaded == m and loaded.verify_signature(b"k")
    tampered = BundleManifest.from_dict({**m.to_dict(), "trie_sha256": "evil"})
    assert not tampered.verify_signature(b"k")
