"""Training objectives. §8.1, §8.2, §8.4.

    L = L_seq + lambda_rank * L_rank + lambda_reg * ||theta - theta_prev||^2
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_LEVEL_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 0.5)
DEFAULT_MARGINS = (0.5, 0.5, 1.0, 1.0, 1.0)


def seq_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    level_weights=DEFAULT_LEVEL_WEIGHTS,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Stage A. ``logits [B, L, 256]``, ``targets [B, L]`` codes, ``weights [B]`` family weights.

    The terminal ordinal is downweighted (alpha_5 = 0.5): it is memorization, and
    over-weighting it burns capacity that belongs to the routing levels. Label smoothing
    applies to levels 1..4 only.
    """
    B, L, K = logits.shape
    per_level = []
    for level in range(L):
        eps = label_smoothing if level < 4 else 0.0
        per_level.append(F.cross_entropy(logits[:, level].float(), targets[:, level], label_smoothing=eps,
                                         reduction="none"))
    per_example = (torch.stack(per_level, 1) * logits.new_tensor(level_weights[:L])).sum(1)
    if weights is not None:
        return (per_example * weights).sum() / weights.sum().clamp_min(1e-9)
    return per_example.mean()


def divergence_level(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """First level (0-based) where each negative's code differs from the positive's.
    ``pos [B, L]``, ``neg [B, N, L]`` -> ``[B, N]``; ``L`` if identical."""
    differs = neg != pos[:, None, :]
    L = pos.shape[1]
    first = torch.where(differs, torch.arange(L, device=pos.device), L)
    return first.min(-1).values


def rank_loss(
    logits: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    neg_mask: torch.Tensor,
    margins=DEFAULT_MARGINS,
) -> torch.Tensor:
    """Stage B/C prefix-oriented margin loss (RIPOR-style).

        L_rank = sum_q sum_i sum_{d- in Neg(q,i)} max(0, m_i - s(q, c<=i^+) + s(q, c<=i^-))

    A negative in ``Neg(q, i)`` shares the positive's first ``i-1`` codes, so the two
    prefix scores differ only in the level-``i`` term, and both terms come from the same
    distribution: the one the teacher-forced positive pass already computed. So one
    forward pass prices every negative, whatever its source (prefix sibling, BM25,
    self-negative). That is the quantity beam search actually consumes: whether the right
    branch outranks its siblings at the step where they compete.

    ``logits [B, L, 256]`` teacher-forced on ``pos [B, L]``; ``neg [B, N, L]`` codes;
    ``neg_mask [B, N]`` marks real (non-padding) negatives.
    """
    lp = logits.float().log_softmax(-1)
    level = divergence_level(pos, neg)  # [B, N]
    valid = neg_mask & (level < pos.shape[1])
    level_c = level.clamp_max(pos.shape[1] - 1)
    lp_at = lp.gather(1, level_c[..., None].expand(-1, -1, lp.shape[-1]))  # [B, N, 256]
    s_pos = lp_at.gather(2, pos.gather(1, level_c)[..., None]).squeeze(-1)
    s_neg = lp_at.gather(2, neg.gather(2, level_c[..., None]).squeeze(-1)[..., None]).squeeze(-1)
    m = logits.new_tensor(margins).float()[level_c]
    hinge = F.relu(m - s_pos + s_neg) * valid
    return hinge.sum() / valid.sum().clamp_min(1)


def proximal(params, reference: list[torch.Tensor]) -> torch.Tensor:
    """``||theta - theta_prev||^2``. Active only in adapter refresh (§15), where drift from
    the deployed checkpoint is itself a risk; zero in full training."""
    return sum(((p - r) ** 2).sum() for p, r in zip(params, reference, strict=True))
