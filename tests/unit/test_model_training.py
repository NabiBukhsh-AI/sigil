"""§33 model row: loss decreases, every level's head gets gradient, the model can overfit a
small corpus, and LoRA leaves the encoder untouched. Uses a tiny encoder, no download."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sigil_core.ids import SemanticId  # noqa: E402
from sigil_model.config import ModelConfig  # noqa: E402
from sigil_model.encoder_decoder import ModelScorer, SigilModel  # noqa: E402
from sigil_model.lora import adapter_state_dict, apply_lora  # noqa: E402
from sigil_training.loop import LoopConfig, collate, train  # noqa: E402
from sigil_training.objectives import divergence_level, rank_loss, seq_loss  # noqa: E402
from sigil_training.replay import stratified_replay  # noqa: E402

pytestmark = pytest.mark.torch
D = 32


class TinyEncoder(torch.nn.Module):
    def __init__(self, vocab=64):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, D)
        self.mix = torch.nn.Linear(D, D)

    def forward(self, input_ids, attention_mask):
        class Out:
            pass

        o = Out()
        o.last_hidden_state = torch.tanh(self.mix(self.emb(input_ids)))
        return o


class CharTok:
    """Whitespace-free char tokenizer shaped like a HF tokenizer call."""

    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="pt"):
        texts = [texts] if isinstance(texts, str) else texts
        ids = [[ord(c) % 63 + 1 for c in t][:max_length] for t in texts]
        n = max(map(len, ids))
        input_ids = torch.tensor([x + [0] * (n - len(x)) for x in ids])
        mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids])

        class Enc(dict):
            def to(self, device):
                return Enc({k: v.to(device) for k, v in self.items()})

        return Enc(input_ids=input_ids, attention_mask=mask)


def tiny_model():
    cfg = ModelConfig(d_model=D, decoder_layers=2, decoder_heads=4, decoder_ff=64, dropout=0.0)
    return SigilModel(cfg, encoder=TinyEncoder())


def rows(n=24, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        codes = [int(c) for c in rng.integers(0, 6, 4)] + [int(rng.integers(0, 3))]
        out.append({"input_text": f"query: doc {i} about topic {codes[0]}", "target_codes": codes,
                    "source": "synthetic_query", "weight": 1.0})
    return out


def test_every_level_head_gets_gradient():
    m = tiny_model()
    b = collate(rows(8), CharTok(), 64, 4, "cpu")
    seq_loss(m(b["input_ids"], b["attention_mask"], b["tokens"]), b["codes"]).backward()
    g = m.embed.weight.grad
    for level in range(5):
        start = 8 + level * 256
        assert g[start : start + 256].abs().sum() > 0, f"level {level + 1} head got no gradient"


def test_overfits_small_corpus():
    torch.manual_seed(0)
    m = tiny_model()
    data = rows(24)
    cfg = LoopConfig(steps=400, lr=3e-3, lr_backbone=3e-3, warmup=20, schedule="constant", batch_size=24,
                     label_smoothing=0.0, log_every=50)
    hist = train(m, iter(lambda: data, None), CharTok(), cfg, device="cpu")
    assert hist[-1]["loss"] < hist[0]["loss"] * 0.2
    b = collate(data, CharTok(), 64, 4, "cpu")
    with torch.no_grad():
        pred = m(b["input_ids"], b["attention_mask"], b["tokens"]).argmax(-1)
    assert (pred == b["codes"]).all(1).float().mean() > 0.9  # exact_id_accuracy


def test_rank_loss_uses_divergence_level():
    pos = torch.tensor([[1, 2, 3, 4, 0]])
    neg = torch.tensor([[[1, 2, 9, 4, 0], [5, 2, 3, 4, 0], [1, 2, 3, 4, 0]]])
    assert divergence_level(pos, neg).tolist() == [[2, 0, 5]]
    logits = torch.zeros(1, 5, 256)
    logits[0, 2, 3] = 10.0  # positive dominates at level 3
    logits[0, 0, 5] = 10.0  # negative dominates at level 1
    loss = rank_loss(logits, pos, neg, torch.tensor([[True, True, True]]))
    assert loss > 0  # driven by the level-1 negative; the identical one is ignored


def test_lora_freezes_encoder_and_trains_adapters():
    m = tiny_model()
    enc_before = {k: v.clone() for k, v in m.encoder.state_dict().items()}
    params = apply_lora(m, rank=4, alpha=8)
    assert params and all(p.requires_grad for p in params)
    assert not any(p.requires_grad for p in m.encoder.parameters())
    cfg = LoopConfig(steps=5, lr=1e-2, warmup=1, schedule="constant", sam=True, log_every=1)
    train(m, iter(lambda: rows(8), None), CharTok(), cfg, device="cpu")
    assert all(torch.equal(v, m.encoder.state_dict()[k]) for k, v in enc_before.items())
    assert any(v.abs().sum() > 0 for k, v in adapter_state_dict(m).items() if k.endswith(".B"))


def test_model_scorer_plugs_into_decoding(tmp_path):
    from sigil_decoding import DecodeConfig, decode
    from sigil_trie import TrieSnapshot, write

    ids = [SemanticId.from_ordinal((a, b, 0, 0), 0) for a in range(3) for b in range(3)]
    path, sha = write(ids, tmp_path, id_schema="ids_v1", corpus_snapshot="cs_2026_09_01")
    with TrieSnapshot(path, sha) as trie:
        r = decode(ModelScorer(tiny_model(), CharTok()), "anything", trie, DecodeConfig(beam=4))
        assert r.candidates and all(c.sid in trie for c in r.candidates)


def test_replay_is_stratified_over_level1():
    old = [{"target_codes": [0, 0, 0, 0, 0]}] * 900 + [{"target_codes": [c, 0, 0, 0, 0]} for c in range(1, 10) for _ in range(30)]
    got = stratified_replay(old, n_new=40, ratio=3.0)
    assert len(got) == 120
    counts = np.bincount([r["target_codes"][0] for r in got])
    assert counts.max() <= 13  # the popular stratum does not dominate
