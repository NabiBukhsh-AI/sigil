"""Shared test data builders."""

import zlib

import numpy as np
from sigil_eval.baselines.router import HashEmbedder, RouterScorer, overlap_reranker  # noqa: F401

hash_embed = HashEmbedder(64)


def clustered_embeddings(n: int, dim: int = 32, clusters: int = 40, seed: int = 0) -> np.ndarray:
    """Clumpy synthetic embeddings: real corpora are heavy-tailed, so cluster sizes are too."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim))
    sizes = rng.zipf(1.6, size=n) % clusters
    return (centers[sizes] + 0.35 * rng.normal(size=(n, dim))).astype(np.float32)


class ToyScorer:
    """Deterministic stand-in for the model: noisy logits with a bump on the gold path.
    Implements the ``sigil_decoding.Scorer`` protocol."""

    def __init__(self, gold: dict, strength: float = 3.0, noise: float = 1.0):
        self.gold, self.strength, self.noise = gold, strength, noise

    def encode(self, q):
        return q

    def step(self, q, prefixes):
        g = self.gold[q]
        level = prefixes.shape[1]
        out = np.empty((len(prefixes), 256))
        for i, p in enumerate(prefixes):
            seed = zlib.crc32(f"{q}|{bytes(p.astype(np.uint8)).hex()}".encode())
            out[i] = np.random.default_rng(seed).normal(size=256) * self.noise
        target = g.codes[level] if level < 4 else g.u
        out[(prefixes == np.array(g.codes[:level])).all(1), target] += self.strength
        return out


# -- tiny torch model for pipeline smoke tests (no downloads) ---------------------------------

D = 32


def tiny_model():
    import torch
    from sigil_model.config import ModelConfig
    from sigil_model.encoder_decoder import SigilModel

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

    cfg = ModelConfig(d_model=D, decoder_layers=2, decoder_heads=4, decoder_ff=64, dropout=0.0)
    return SigilModel(cfg, encoder=TinyEncoder())


class CharTok:
    """Char tokenizer shaped like a HF tokenizer call."""

    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="pt"):
        import torch

        texts = [texts] if isinstance(texts, str) else texts
        ids = [[ord(c) % 63 + 1 for c in t][:max_length] for t in texts]
        n = max(map(len, ids))
        input_ids = torch.tensor([x + [0] * (n - len(x)) for x in ids])
        mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids])

        class Enc(dict):
            def to(self, device):
                return Enc({k: v.to(device) for k, v in self.items()})

        return Enc(input_ids=input_ids, attention_mask=mask)
