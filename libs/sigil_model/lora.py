"""LoRA adapters for incremental refresh. §15.5.

[FIXED] Decoder self-attention, cross-attention, and FFN only. The encoder holds general
query understanding and must not be perturbed by a small batch of new documents, so it is
frozen along with everything else that is not an adapter.
"""

from __future__ import annotations

import math

import torch
from torch import nn

TARGETS = {
    "decoder_self_attn": "self_attn.",
    "decoder_cross_attn": "cross_attn.",
    "decoder_ffn": "ffn_",
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))  # zero: starts as identity
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.T @ self.B.T) * self.scale


def apply_lora(model: nn.Module, rank: int = 16, alpha: float = 32, dropout: float = 0.05,
               target_modules: list[str] | tuple[str, ...] = tuple(TARGETS)) -> list[nn.Parameter]:
    """Wrap targeted decoder linears in place, freeze everything else, return adapter params."""
    needles = [TARGETS[t] for t in target_modules]
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in list(model.named_modules()):
        if not name.startswith("layers."):
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(n in f"{name}.{child_name}" for n in needles):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
    return [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}


def load_adapter(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing = set(state) - {n for n, _ in model.named_parameters()}
    if missing:
        raise KeyError(f"adapter keys not in model (apply_lora first?): {sorted(missing)[:3]}")
    model.load_state_dict(state, strict=False)
