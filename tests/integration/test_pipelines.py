"""Pipeline smoke: dataset build -> stages A to D -> offline gates -> signed bundle ->
verify_bundle -> adapter refresh. Tiny model, a few steps; this proves the wiring, not
quality."""

import pytest

torch = pytest.importorskip("torch")

from sigil_core.config import load_yaml  # noqa: E402
from sigil_identifiers.quantizer import codebook_io  # noqa: E402
from sigil_trie import TrieSnapshot  # noqa: E402

from pipelines import adapter_refresh, full_train  # noqa: E402
from pipelines.dataset_build import build_rows  # noqa: E402
from scripts.local_stack import build, query_for, synthetic_corpus  # noqa: E402
from scripts.verify_bundle import verify  # noqa: E402
from tests.helpers import CharTok, tiny_model  # noqa: E402

pytestmark = pytest.mark.torch
LOOSE = {**load_yaml("configs/gates/release_gates.yaml")["gates"], "held_out_doc_recall_at_10_min": 0.0,
         "ece_max": 1.0, "escape_rate_max": 1.0}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    docs = synthetic_corpus(4, 8)
    s = build(docs, tmp / "stack")
    text = {i: t for _, i, t in docs}
    corpus = []
    for i, parent in s.parent.items():
        r = s.reg.children(parent)[0]
        corpus.append({"doc_uid": r.doc_uid, "semantic_id": str(r.semantic_id), "title": r.title, "text": text[i]})
    synthetic = {c["doc_uid"]: [{"text": f"unique{i}{x} topic{t}word{j}"} for x in "abc" for j in range(4)]
                 for c, (t, i, _) in zip(corpus, docs, strict=True)}
    cfg = load_yaml("configs/model/base.yaml")
    # 32 docs at the default 5% would leave the held-out slice empty in ~19% of runs (doc
    # uids are random); 25% makes an empty slice a ~1e-4 event.
    cfg["data"]["held_out"]["document_fraction"] = 0.25
    rows, rep = build_rows(corpus, synthetic, [], cfg, corpus_snapshot="cs_2026_09_01", id_schema="ids_v1")
    golden = [(query_for(t, i), {s.reg.children(s.parent[i])[0].doc_uid}) for t, i, _ in docs]
    cb_dir = tmp / "codebooks"
    codebook_io.save(s.pipe.q.codebooks, cb_dir, version="cb_v1.0", id_schema="ids_v1")
    info = s.last_snapshot
    s.close()
    return tmp, cfg, rows, rep, corpus, golden, cb_dir, info


def test_full_train_writes_a_verifiable_signed_bundle(setup):
    tmp, cfg, rows, rep, corpus, golden, cb_dir, info = setup
    torch.manual_seed(0)
    report = full_train.train_bundle(
        model=tiny_model(), tokenizer=CharTok(), rows=rows, cfg=cfg, trie_path=info["path"], trie_sha256=info["sha256"],
        codebooks_dir=str(cb_dir), corpus=corpus, golden=golden, held_out=set(rep["held_out_doc_ids"]),
        out_dir=str(tmp / "bundles"), bundle_id="bundle_smoke", backbone="tiny", dataset="ds_smoke",
        corpus_snapshot="cs_2026_09_01", epoch_range=(0, 10**9), signing_key=b"k", steps=15, batch_size=16,
        device="cpu", thresholds=LOOSE,
    )
    assert report["metrics"]["valid_id_rate"] == 1.0
    assert report["offline_gates"]["passed"], report["offline_gates"]["summary"]
    assert verify(report["bundle_dir"], b"k") == []
    with pytest.raises(Exception):
        verify(report["bundle_dir"], b"wrong-key")


def test_default_gates_block_an_untrained_model(setup, tmp_path):
    _, cfg, rows, rep, corpus, golden, cb_dir, info = setup
    report = full_train.train_bundle(
        model=tiny_model(), tokenizer=CharTok(), rows=rows, cfg=cfg, trie_path=info["path"], trie_sha256=info["sha256"],
        codebooks_dir=str(cb_dir), corpus=corpus, golden=golden, held_out=set(rep["held_out_doc_ids"]),
        out_dir=str(tmp_path), bundle_id="bundle_blocked", backbone="tiny", dataset="ds_smoke",
        corpus_snapshot="cs_2026_09_01", epoch_range=(0, 10**9), steps=1, batch_size=16, device="cpu",
    )
    assert not report["offline_gates"]["passed"]
    assert not (tmp_path / "bundle_blocked" / "manifest.json").exists()  # a report, never a bundle
    assert (tmp_path / "bundle_blocked" / "eval_report.json").exists()


def test_adapter_refresh_gates_and_keeps_the_encoder_frozen(setup, tmp_path):
    _, cfg, rows, _, corpus, golden, _, info = setup
    model = tiny_model()
    enc = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    new = [r for r in rows if r["doc_id"] in {c["doc_uid"] for c in corpus[:6]}]
    with TrieSnapshot(info["path"], info["sha256"]) as trie:
        decision = adapter_refresh.refresh(
            model=model, tokenizer=CharTok(), trie=trie, new_rows=new, replay_pool=rows, eval_new=golden[:6],
            eval_old=golden[6:], sid_to_doc={c["semantic_id"]: c["doc_uid"] for c in corpus},
            lora_cfg={**cfg["lora"], "rank": 4, "alpha": 8}, lifecycle={"adapter_min_batch": 1, "replay_ratio": 3.0},
            steps=5, out_dir=str(tmp_path / "ad"), adapter_version="ad_smoke", backbone="tiny",
            trained_on_snapshot="cs_2026_09_01", batch_size=8,
        )
    assert decision["decision"] in {"accepted", "rejected"}
    assert (tmp_path / "ad" / "adapter.pt").exists() == (decision["decision"] == "accepted")
    assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in enc.items())
    skipped = adapter_refresh.refresh(model=None, tokenizer=None, trie=None, new_rows=new, replay_pool=[], eval_new=[],
                                      eval_old=[], sid_to_doc={}, lora_cfg={}, lifecycle={"adapter_min_batch": 2000},
                                      steps=1, out_dir="", adapter_version="", backbone="", trained_on_snapshot="")
    assert skipped["decision"] == "skipped"


def test_hot_sweep_waits_seven_days(tmp_path):
    from datetime import UTC, datetime, timedelta

    from sigil_eval.baselines.bm25 import BM25
    from sigil_registry_client import MemoryRegistry

    reg, hot = MemoryRegistry(), BM25()
    r = reg.create(tenant_id="t", codes=(1, 2, 3, 4), content_hash="h", title="x", rerank_snippet="alpha beta")
    hot.add([str(r.semantic_id)], ["alpha beta"])
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    adapter_refresh.mark_learned(reg, [r.doc_uid], t0)
    assert adapter_refresh.sweep_hot_set(reg, hot, t0 + timedelta(days=6)) == 0 and hot.search("alpha")
    assert adapter_refresh.sweep_hot_set(reg, hot, t0 + timedelta(days=7)) == 1 and not hot.search("alpha")
