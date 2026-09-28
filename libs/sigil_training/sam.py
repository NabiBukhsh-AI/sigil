"""Sharpness-aware minimization. §8.7.

DSI++ used SAM to flatten loss basins and so reduce forgetting during continual indexing.
It roughly doubles step cost: optional in stage A, mandatory in adapter refresh.
"""

from __future__ import annotations

import torch


class SAM:
    """Wraps a base optimizer. Usage per step::

        loss_fn().backward(); sam.first_step()
        loss_fn().backward(); sam.second_step()
    """

    def __init__(self, params, base: torch.optim.Optimizer, rho: float = 0.05):
        self.params = [p for p in params if p.requires_grad]
        self.base, self.rho = base, rho
        self._eps: list[torch.Tensor | None] = []

    @torch.no_grad()
    def first_step(self) -> None:
        grads = [p.grad for p in self.params if p.grad is not None]
        norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2) if grads else torch.tensor(0.0)
        scale = self.rho / (norm + 1e-12)
        self._eps = []
        for p in self.params:
            if p.grad is None:
                self._eps.append(None)
                continue
            e = p.grad * scale
            p.add_(e)
            self._eps.append(e)
        self.base.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self) -> None:
        for p, e in zip(self.params, self._eps, strict=True):
            if e is not None:
                p.sub_(e)
        self.base.step()
        self.base.zero_grad(set_to_none=True)
