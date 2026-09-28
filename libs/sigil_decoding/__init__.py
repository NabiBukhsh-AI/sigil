"""Constrained decoding: beam search, terminal fan-out, diversity, confidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sigil_decoding.beam import BeamOutput, DecodeConfig, Scorer, beam_search
from sigil_decoding.confidence import Calibrator
from sigil_decoding.constraints import MaskProvider
from sigil_decoding.fanout import Candidate, terminal_fanout


@dataclass
class DecodeResult:
    candidates: list[Candidate]
    beam: BeamOutput
    s_top: float
    margin_1: float
    confidence: float  # calibrated P(top result relevant); routing only, never ranking


def decode(
    scorer: Scorer,
    query_or_state: Any,
    trie: MaskProvider,
    cfg: DecodeConfig,
    calibrator: Calibrator = Calibrator(),
    *,
    encoded: bool = False,
) -> DecodeResult:
    """One query, one pass: encode once, beam over 4 levels, fan out the terminal level."""
    state = query_or_state if encoded else scorer.encode(query_or_state)
    beam = beam_search(scorer, state, trie, cfg)
    cands = terminal_fanout(scorer, state, trie, beam, cfg)
    s_top = cands[0].score if cands else float("-inf")
    conf = calibrator.prob(s_top) if cands else 0.0
    return DecodeResult(cands, beam, s_top, beam.margin_1, conf)


__all__ = ["Candidate", "DecodeConfig", "DecodeResult", "Scorer", "beam_search", "decode"]
