"""Encoder-decoder with an identifier-only decoder vocabulary. §4.2, §4.3, §6.

[FIXED] Asymmetric split: the backbone's 12-layer encoder reads the query once and its
full token-state matrix ``H`` is kept (not pooled), so the identifier decision
cross-attends to token-level query structure. A shallow 4-layer decoder emits exactly 5
level-scoped tokens; it never shares the text vocabulary, so it cannot emit text and pays
a 256-way softmax per step instead of 32k.

The decoder is a small pre-LN stack written out explicitly (named q/k/v/o and FFN
linears) so LoRA adapters (§15.5) can target it by name. Positions are learned absolute
over 6 slots; relative position bias is pointless at length 5.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sigil_core.ids import BOS_ID, TOKEN_LEVELS, VOCAB_SIZE, K, token_id
from sigil_model.config import ModelConfig
from sigil_model.heads import PerLevelHeads


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.dropout = heads, dropout
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x, kv, mask=None, causal=False):
        B, T, d = x.shape
        S = kv.shape[1]

        def split(t, n):
            return t.view(B, n, self.h, d // self.h).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            split(self.q(x), T), split(self.k(kv), S), split(self.v(kv), S),
            attn_mask=mask, is_causal=causal, dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o(out.transpose(1, 2).reshape(B, T, d))


class DecoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.self_attn = Attention(d, heads, dropout)
        self.cross_attn = Attention(d, heads, dropout)
        self.ffn_in, self.ffn_out = nn.Linear(d, ff), nn.Linear(ff, d)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, H, enc_mask):
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, causal=True))
        x = x + self.drop(self.cross_attn(self.ln2(x), H, mask=enc_mask))
        return x + self.drop(self.ffn_out(self.drop(F.gelu(self.ffn_in(self.ln3(x))))))


class SigilModel(nn.Module):
    def __init__(self, cfg: ModelConfig, encoder: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        if encoder is None:
            from transformers import T5EncoderModel

            encoder = T5EncoderModel.from_pretrained(cfg.backbone_init)
        self.encoder = encoder
        d = cfg.d_model
        self.embed = nn.Embedding(VOCAB_SIZE, d)
        self.pos = nn.Embedding(TOKEN_LEVELS + 1, d)
        self.layers = nn.ModuleList(
            DecoderLayer(d, cfg.decoder_heads, cfg.decoder_ff, cfg.dropout) for _ in range(cfg.decoder_layers)
        )
        self.ln_f = nn.LayerNorm(d)
        self.heads = PerLevelHeads(self.embed)
        nn.init.normal_(self.embed.weight, std=d**-0.5)

    # -- pieces ---------------------------------------------------------------------------

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return H, attention_mask.bool()[:, None, None, :]

    def decode_hidden(self, tokens: torch.Tensor, H: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        """``tokens [B, t]`` starting with BOS -> hidden ``[B, t, d]``. Causal."""
        t = tokens.shape[1]
        x = self.embed(tokens) + self.pos(torch.arange(t, device=tokens.device))
        for layer in self.layers:
            x = layer(x, H, enc_mask)
        return self.ln_f(x)

    # -- training -------------------------------------------------------------------------

    def forward(self, input_ids, attention_mask, target_tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits ``[B, L, 256]`` for ``target_tokens [B, L]`` (L <= 5)."""
        H, m = self.encode(input_ids, attention_mask)
        bos = torch.full_like(target_tokens[:, :1], BOS_ID)
        hidden = self.decode_hidden(torch.cat([bos, target_tokens[:, :-1]], 1), H, m)
        return self.heads.all_levels(hidden)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def codes_to_tokens(prefixes: np.ndarray) -> np.ndarray:
    """``[n, l]`` codes -> level-scoped token ids."""
    lvl = np.arange(1, prefixes.shape[1] + 1)
    return (token_id(1, 0) + (lvl - 1) * K + prefixes).astype(np.int64)


class ModelScorer:
    """Adapts ``SigilModel`` to the ``sigil_decoding.Scorer`` protocol.

    ponytail: recomputes the decoder over the whole prefix each step instead of caching
    KV. The prefix is at most 5 tokens over a 4-layer decoder; add a KV cache if decode
    shows up in profiles.
    """

    def __init__(self, model: SigilModel, tokenizer, device: str = "cpu", tenant_id: str | None = None):
        from sigil_model.tokenization import format_query

        self.model, self.tok, self.device = model.eval(), tokenizer, device
        self._fmt = lambda q: format_query(q, tenant_id)

    def encode(self, query: str):
        return self.encode_text(self._fmt(query))

    @torch.inference_mode()
    def encode_text(self, text: str):
        """Encode an already-formatted input (training rows carry ``input_text``)."""
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.model.cfg.max_query_tokens).to(self.device)
        return self.model.encode(enc["input_ids"], enc["attention_mask"])

    @torch.inference_mode()
    def step(self, state, prefixes: np.ndarray) -> np.ndarray:
        H, m = state
        n, level = len(prefixes), prefixes.shape[1] + 1
        tokens = np.concatenate([np.full((n, 1), BOS_ID), codes_to_tokens(prefixes)], 1)
        tokens = torch.as_tensor(tokens, device=self.device)
        hidden = self.model.decode_hidden(tokens, H.expand(n, -1, -1), m.expand(n, -1, -1, -1))
        return self.model.heads(hidden[:, -1], level).float().cpu().numpy()
