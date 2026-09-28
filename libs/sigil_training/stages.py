"""The four training stages. §8.

A: indexing and retrieval seq2seq objective (L_seq).
B: prefix-oriented ranking (L_seq + lambda_rank * L_rank) over mined negatives.
C: self-negative refinement: decode the training queries with the real constrained beam,
   and anything ranked above the positive becomes a negative. Attacks the errors the
   deployed decoder will actually make.
D: freeze, fit a temperature on held-out data, report ECE before and after.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace

import numpy as np

from sigil_core.ids import SemanticId
from sigil_decoding import DecodeConfig, decode
from sigil_decoding.confidence import fit_temperature
from sigil_training.loop import LoopConfig, train

QUERY_SOURCES = {"synthetic_query", "real_query", "title", "metadata"}


def stage_a(model, batches, tokenizer, cfg: LoopConfig, **kw):
    return train(model, batches, tokenizer, replace(cfg, lambda_rank=0.0), **kw)


def stage_b(model, batches, tokenizer, cfg: LoopConfig, **kw):
    if not cfg.lambda_rank:
        raise ValueError("stage B without lambda_rank is stage A")
    return train(model, batches, tokenizer, cfg, **kw)


def mine_self_negatives(scorer, trie, rows: Iterable[dict], decode_cfg: DecodeConfig, max_per_row: int = 8):
    """Yield rows with ``hard_negative_codes`` extended by every identifier the current
    checkpoint ranks above the true positive. Content-prefix rows are passed through."""
    for row in rows:
        if row.get("source") not in QUERY_SOURCES:
            yield row
            continue
        gold = tuple(row["target_codes"][:4])
        res = decode(scorer, scorer.encode_text(row["input_text"]), trie, decode_cfg, encoded=True)
        above = []
        for c in res.candidates:
            if c.sid.codes == gold and c.sid.u == row["target_codes"][4]:
                break
            above.append([*c.sid.codes, c.sid.u])
        yield {**row, "hard_negative_codes": (row.get("hard_negative_codes") or []) + above[:max_per_row]}


def stage_c(model, make_batches: Callable[[Callable], Iterable], tokenizer, scorer, trie, cfg: LoopConfig,
            decode_cfg: DecodeConfig, rounds: int = 2, **kw):
    """``make_batches(transform)`` returns a fresh batch iterator whose rows have passed
    through ``transform``, so each round mines against the current checkpoint."""
    history = []
    for _ in range(rounds):
        def transform(rows, _s=scorer):
            return list(mine_self_negatives(_s, trie, rows, decode_cfg))

        history += stage_b(model, make_batches(transform), tokenizer, cfg, **kw)
    return history


def stage_d(scorer, trie, heldout: Sequence[tuple[str, set[SemanticId]]], decode_cfg: DecodeConfig):
    """Returns ``(Calibrator, report)``. ``heldout`` is ``(query, relevant ids)`` pairs.

    [HONEST] Label smoothing in stage A degrades calibration measurably; this stage is why
    it is kept rather than dropped.
    """
    from sigil_eval.metrics import ece

    s, y = [], []
    for query, relevant in heldout:
        res = decode(scorer, query, trie, decode_cfg)
        if res.candidates:
            s.append(res.s_top)
            y.append(float(res.candidates[0].sid in relevant))
    s, y = np.array(s), np.array(y)
    before = ece(1 / (1 + np.exp(-s)), y)  # uncalibrated: T=1, b=0
    cal = fit_temperature(s, y)
    after = ece(np.array([cal.prob(v) for v in s]), y)
    return cal, {"ece_before": before, "ece_after": after, "temperature": cal.temperature,
                 "bias": cal.bias, "n": int(len(s))}
