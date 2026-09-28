"""Cross-channel merge, near-duplicate collapse, diversity quota, and score blending.
§11.2, §13.3.

    final(q,d) = w1 * sigma(CE/T) + w2 * norm(s_gen) + w3 * bm25_norm + w4 * prior(d)

Keeping weight on ``s_gen`` matters: it is the only term carrying the model's global view
rather than a pairwise one, and it degrades gracefully when the reranker is down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigil_decoding.diversity import quota_select

from services.gateway.verification import Cand

CHANNEL_ORDER = {"generative": 0, "hot_lexical": 1, "full_lexical": 2}


@dataclass(frozen=True)
class Weights:
    cross_encoder: float = 0.70
    generative: float = 0.15
    bm25: float = 0.10
    prior: float = 0.05

    @classmethod
    def from_cfg(cls, b: dict) -> Weights:
        return cls(b["w_cross_encoder"], b["w_generative"], b["w_bm25"], b["w_prior"])


def preliminary_order(cands: list[Cand]) -> list[Cand]:
    """Before reranking there is no common scale, so order by channel, then by the
    channel's own score."""
    def key(c: Cand):
        s = c.gen if c.channel == "generative" else c.bm25
        return (CHANNEL_ORDER[c.channel], -(s if s is not None else -1e9), str(c.sid))

    return sorted(cands, key=key)


def collapse_near_duplicates(cands: list[Cand]) -> list[Cand]:
    """Keep the best-ranked member of each near-duplicate link group (§7.4)."""
    seen, out = set(), []
    for c in cands:
        g = getattr(c.record, "near_dup_group", None) or c.record.doc_uid
        if g not in seen:
            seen.add(g)
            out.append(c)
    return out


def apply_quota(cands: list[Cand], n: int, quota: float) -> list[Cand]:
    """At most ``quota`` of the list from one level-1 code, backfilled when short."""
    l1 = np.array([c.sid.codes[0] for c in cands])
    return [cands[i] for i in quota_select(l1, n, quota)]


def blend(cands: list[Cand], w: Weights) -> None:
    if not cands:
        return
    gen = np.array([c.gen if c.gen is not None else -np.inf for c in cands])
    gen_norm = np.exp(gen - gen.max()) if np.isfinite(gen).any() else np.zeros(len(cands))
    bm = np.array([c.bm25 or 0.0 for c in cands])
    bm_norm = bm / bm.max() if bm.max() > 0 else bm
    prior = np.array([float(c.record.metadata.get("prior", 0.5)) for c in cands])
    have_ce = all(c.ce is not None for c in cands)
    w_ce = w.cross_encoder if have_ce else 0.0
    total = w_ce + w.generative + w.bm25 + w.prior
    for i, c in enumerate(cands):
        ce = c.ce if have_ce else 0.0
        c.final = float((w_ce * ce + w.generative * gen_norm[i] + w.bm25 * bm_norm[i] + w.prior * prior[i]) / total)
    cands.sort(key=lambda c: (-c.final, str(c.sid)))
