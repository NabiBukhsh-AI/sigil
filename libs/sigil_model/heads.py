"""Per-level output projections, tied to the decoder's input embeddings. §4.3.

Five ``768 x 256`` projections rather than one ``768 x 1280``: identical cost, but a code
integer means different things at different depths and per-level heads stabilize
training. With level-scoped tokens, the head for level ``l`` is exactly the slice of the
embedding table holding level ``l``'s tokens, so the softmax at step ``l`` cannot place
mass on another level's codes even before trie masking.
"""

from __future__ import annotations

import torch
from torch import nn

from sigil_core.ids import BASE, K


class PerLevelHeads(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding  # tied; not a copy
        self.bias = nn.Parameter(torch.zeros(5, K))

    def weight(self, level: int) -> torch.Tensor:
        start = BASE + (level - 1) * K
        return self.embedding.weight[start : start + K]

    def forward(self, hidden: torch.Tensor, level: int) -> torch.Tensor:
        """``hidden [..., d]`` at the position predicting ``level`` -> ``[..., 256]``."""
        return hidden @ self.weight(level).T + self.bias[level - 1]

    def all_levels(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden [B, L, d]`` where position ``i`` predicts level ``i+1`` -> ``[B, L, 256]``."""
        return torch.stack([self(hidden[:, i], i + 1) for i in range(hidden.shape[1])], dim=1)
