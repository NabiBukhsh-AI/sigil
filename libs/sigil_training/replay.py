"""Stratified replay for adapter refresh. §15.5.

Replay of existing documents' pseudo-queries is what prevents forgetting (DSI++). The
buffer is stratified over level-1 codes so replay covers the identifier space rather than
the popular regions: every level-1 stratum gets an equal share, and strata too small to
fill their share hand the remainder to the others.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def stratified_replay(old: Sequence[dict], n_new: int, ratio: float = 3.0, seed: int = 0) -> list[dict]:
    """Sample ``ratio * n_new`` rows from ``old`` (rows carry ``target_codes``)."""
    want = min(int(ratio * n_new), len(old))
    strata: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(old):
        strata[int(row["target_codes"][0])].append(i)
    rng = np.random.default_rng(seed)
    pools = {c: list(rng.permutation(ix)) for c, ix in sorted(strata.items())}
    picked: list[int] = []
    while len(picked) < want:
        live = [c for c in pools if pools[c]]
        share = max(1, (want - len(picked)) // len(live))
        for c in live:
            take, pools[c] = pools[c][:share], pools[c][share:]
            picked.extend(take)
            if len(picked) >= want:
                break
    return [old[i] for i in picked[:want]]
