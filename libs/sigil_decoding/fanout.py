"""Terminal-level fan-out. §10.2 lines 10-15, ADR 0002.

The decoder generates four routing levels. The terminal level is resolved by enumerating
every document the trie holds under each surviving 4-level prefix. Leaves are scored as
``prefix_score + log p(u | prefix, q)`` with ``u`` masked to the ordinals that exist, and
the reranker makes the real decision. A document added after the last training run is
reachable here the moment it is in the trie: its ``u`` was never learned, and it does not
need to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sigil_core.ids import K, SemanticId
from sigil_decoding.beam import BeamOutput, DecodeConfig, Scorer
from sigil_decoding.constraints import MaskProvider, masked_log_softmax
from sigil_decoding.diversity import rank_within_group
from sigil_trie.mmap_reader import tail_tuple


@dataclass(frozen=True)
class Candidate:
    sid: SemanticId
    score: float  # full sequence log-prob: 4 routing levels + u
    prefix_score: float
    prefix_rank: int


def candidate_cap(cfg: DecodeConfig, beam: BeamOutput) -> int:
    """§35B R6: easy queries do not need 100 rerank pairs."""
    if not cfg.adaptive_candidate_cap or len(beam.scores) < 2:
        return cfg.candidate_cap
    separated = beam.scores[0] - beam.scores[1] >= cfg.adaptive_candidate_cap_margin
    if beam.margin_1 >= cfg.adaptive_candidate_cap_margin and separated:
        return min(cfg.candidate_cap, cfg.adaptive_candidate_cap_min)
    return cfg.candidate_cap


def terminal_fanout(
    scorer: Scorer, state: Any, trie: MaskProvider, beam: BeamOutput, cfg: DecodeConfig
) -> list[Candidate]:
    p = min(cfg.prefixes_expanded, len(beam.scores))
    if p == 0:
        return []
    tails, owner = trie.leaves(beam.nodes[:p])
    u = tails[:, 0].astype(np.int64)
    mask = np.zeros((p, K), dtype=bool)
    mask[owner, u] = True
    lpu = masked_log_softmax(scorer.step(state, beam.prefixes[:p]), mask)
    score = beam.scores[:p][owner] + lpu[owner, u]

    order = np.lexsort((*tails.T[::-1], owner, -score))
    keep = order[rank_within_group(owner[order]) < cfg.leaves_per_prefix][: candidate_cap(cfg, beam)]
    return [
        Candidate(
            sid=SemanticId(tuple(int(c) for c in beam.prefixes[owner[i]]), tail_tuple(tails[i])),
            score=float(score[i]),
            prefix_score=float(beam.scores[owner[i]]),
            prefix_rank=int(owner[i]),
        )
        for i in keep
    ]
