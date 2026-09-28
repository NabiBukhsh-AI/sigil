"""Trie-constrained beam search over the four routing levels. §10.1, §10.2.

[FIXED] No sampling of any kind. Every identifier has the same length, so no length
normalization is applied and the usual beam length bias does not exist (§4.4). Ties are
broken by beam index then code, so identical inputs give bitwise-identical beams.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Protocol

import numpy as np

from sigil_core.ids import LEVELS
from sigil_decoding.confidence import entropy, margin
from sigil_decoding.constraints import MaskProvider, assert_in_mask, masked_log_softmax
from sigil_decoding.diversity import quota_select


class Scorer(Protocol):
    """The model surface decoding needs. ``sigil_model`` implements it with the
    encoder-decoder; tests and the brute-force oracle can implement it directly."""

    def encode(self, query: str) -> Any: ...

    def step(self, state: Any, prefixes: np.ndarray) -> np.ndarray:
        """``prefixes`` is ``[n, l]`` codes decoded so far (``l`` in 0..4). Returns raw
        ``[n, 256]`` logits for level ``l + 1``."""
        ...


@dataclass(frozen=True)
class DecodeConfig:
    beam: int = 64
    expansion: int = 2
    prefixes_expanded: int = 32
    leaves_per_prefix: int = 8
    candidate_cap: int = 100
    adaptive_candidate_cap: bool = True
    adaptive_candidate_cap_min: int = 30
    adaptive_candidate_cap_margin: float = 2.5
    quota_l1: float = 0.4
    disable_quota_above_margin: float = 4.0
    adaptive_beam: bool = True
    widen_below_margin: float = 0.5
    widened_beam: int = 128
    multi_start: bool = False
    forced_l1_codes: int = 3

    @classmethod
    def from_yaml_dict(cls, d: dict) -> DecodeConfig:
        """Flatten the ``decoding:`` block of ``configs/decoding/*.yaml``."""
        dec = d.get("decoding", d)
        fo, div = dec.get("terminal_fanout", {}), dec.get("diversity", {})
        ab, ms = dec.get("adaptive_beam", {}), dec.get("multi_start", {})
        flat = {
            **{k: dec[k] for k in ("beam", "expansion", "candidate_cap", "adaptive_candidate_cap",
                                   "adaptive_candidate_cap_min", "adaptive_candidate_cap_margin") if k in dec},
            "prefixes_expanded": fo.get("prefixes_expanded"),
            "leaves_per_prefix": fo.get("leaves_per_prefix"),
            "quota_l1": div.get("quota_l1"),
            "disable_quota_above_margin": div.get("disable_above_margin"),
            "adaptive_beam": ab.get("enabled"),
            "widen_below_margin": ab.get("widen_below_margin"),
            "widened_beam": ab.get("widened_beam"),
            "multi_start": ms.get("enabled"),
            "forced_l1_codes": ms.get("forced_l1_codes"),
        }
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in flat.items() if v is not None and k in known})

    def clamp(self, max_beam: int, max_candidate_cap: int) -> DecodeConfig:
        """§27: server-side caps override client options."""
        return replace(self, beam=min(self.beam, max_beam), widened_beam=min(self.widened_beam, max_beam),
                       candidate_cap=min(self.candidate_cap, max_candidate_cap))


@dataclass
class LevelTrace:
    level: int
    top: list[tuple[str, float]]
    entropy: float
    kept: int
    pruned: int


@dataclass
class BeamOutput:
    prefixes: np.ndarray  # [n, 4] codes, best first
    scores: np.ndarray  # [n] cumulative log-prob over 4 levels
    nodes: np.ndarray  # [n] depth-4 trie node ids
    margin_1: float
    beam_width: int
    levels: list[LevelTrace] = field(default_factory=list)
    survivors: list[np.ndarray] = field(default_factory=list)  # prefixes kept after each level


def beam_search(scorer: Scorer, state: Any, trie: MaskProvider, cfg: DecodeConfig) -> BeamOutput:
    nodes = np.array([trie.ROOT], dtype=np.int64)
    prefixes = np.zeros((1, 0), dtype=np.int64)
    scores = np.zeros(1)
    width, quota, margin_1 = cfg.beam, cfg.quota_l1, 0.0
    trace: list[LevelTrace] = []
    survivors: list[np.ndarray] = []

    for depth in range(LEVELS):
        mask = trie.children_mask(nodes, depth)
        lp = masked_log_softmax(scorer.step(state, prefixes), mask)
        if depth == 0:
            # §10.6 mitigations are decided once, from the routing certainty at level 1.
            margin_1 = margin(lp[0])
            if cfg.adaptive_beam and margin_1 < cfg.widen_below_margin:
                width = max(width, cfg.widened_beam)
            if margin_1 > cfg.disable_quota_above_margin:
                quota = 1.0  # narrow intent: diversity would only dilute the answer
            if cfg.multi_start:
                # ponytail: multi-start as a 1/n per-l1 quota with backfill. Spends ~B/n on
                # each of the top n level-1 codes; exact reservation if the A/B wants it.
                quota = min(quota, 1.0 / cfg.forced_l1_codes)

        total = scores[:, None] + lp
        rows, codes = np.nonzero(np.isfinite(total))
        cand = total[rows, codes]
        # §10.2: the diversity cap draws from the top B*expansion pool, so `expansion` is
        # what gives the quota room. A sharply peaked query can fill the whole pool from one
        # level-1 branch, and then the quota backfills from that branch by design.
        order = np.lexsort((codes, rows, -cand))[: width * cfg.expansion]
        rows, codes, cand = rows[order], codes[order], cand[order]
        keep = quota_select(codes if depth == 0 else prefixes[rows, 0], width, quota)
        rows, codes, cand = rows[keep], codes[keep], cand[keep]
        assert_in_mask(mask, rows, codes)

        prefixes = np.concatenate([prefixes[rows], codes[:, None]], axis=1)
        nodes = trie.child(nodes[rows], depth, codes)
        trace.append(LevelTrace(
            level=depth + 1,
            top=[(".".join(map(str, p)), round(float(s), 4)) for p, s in zip(prefixes[:5], cand[:5], strict=False)],
            entropy=round(entropy(lp[0]), 4),
            kept=len(keep),
            pruned=int(np.isfinite(total).sum()) - len(keep),
        ))
        scores = cand
        survivors.append(prefixes)

    return BeamOutput(prefixes, scores, nodes, margin_1, width, trace, survivors)
