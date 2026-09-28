"""Family ratios, per-unit balancing, long-tail upsampling. §7.2, §7.4.

Families and starting ratios: synthetic 60%, real logs 15% (weight 3x, up to 40% once
available), titles 10%, content prefix 10%, metadata 5%.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

FAMILY_OF_SOURCE = {
    "synthetic_query": "synthetic", "real_query": "logs", "title": "titles",
    "content_prefix": "content_prefix", "metadata": "metadata",
}
LOG_WEIGHT = 3.0


def balance_units(rows: Sequence[dict], max_per_unit: int = 20, zero_log_upsample: float = 1.5,
                  seed: int = 0) -> list[dict]:
    """Cap each unit at ``max_per_unit`` and upsample units with no real-query coverage.
    Without this, head documents dominate and the tail becomes unretrievable."""
    rng = np.random.default_rng(seed)
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_unit[r["doc_id"]].append(r)
    out = []
    for unit in sorted(by_unit):
        # Real query logs are the only true query distribution: they survive the cap first.
        logs = [r for r in by_unit[unit] if r["source"] == "real_query"][:max_per_unit]
        rest = [r for r in by_unit[unit] if r["source"] != "real_query"]
        room = max_per_unit - len(logs)
        if len(rest) > room:
            rest = [rest[i] for i in sorted(rng.choice(len(rest), room, replace=False))]
        rs = logs + rest
        has_logs = bool(logs)
        for r in rs:
            w = r.get("weight", 1.0) * (LOG_WEIGHT if r["source"] == "real_query" else 1.0)
            if not has_logs:
                w *= zero_log_upsample
            out.append({**r, "weight": w})
    return out


def mix(rows: Sequence[dict], ratios: Mapping[str, float]) -> list[dict]:
    """Reweight families so each carries its target share of total loss weight.

    Rows are never dropped: the per-unit coverage floor is [FIXED] (every unit keeps at
    least ``min_queries_per_doc`` examples) while the ratios are [TUNE]. Subsampling to hit
    a ratio would trade the first for the second. Families absent from the data (no real
    logs yet) have their share renormalized over the families present.
    """
    total_w: dict[str, float] = defaultdict(float)
    for r in rows:
        total_w[FAMILY_OF_SOURCE[r["source"]]] += r.get("weight", 1.0)
    present = {f: ratios.get(f, 0.0) for f in total_w if ratios.get(f, 0.0) > 0}
    norm = sum(present.values()) or 1.0
    grand = sum(total_w.values())
    scale = {f: (present.get(f, 0.0) / norm) * grand / total_w[f] for f in total_w}
    return [{**r, "weight": r.get("weight", 1.0) * scale[FAMILY_OF_SOURCE[r["source"]]]} for r in rows]


def family_shares(rows: Sequence[dict]) -> dict[str, float]:
    w: dict[str, float] = defaultdict(float)
    for r in rows:
        w[FAMILY_OF_SOURCE[r["source"]]] += r.get("weight", 1.0)
    total = sum(w.values()) or 1.0
    return {f: v / total for f, v in w.items()}


def coverage(rows: Sequence[dict]) -> dict[str, int]:
    """Examples per unit, for the ``min_queries_per_doc`` floor check."""
    c: dict[str, int] = defaultdict(int)
    for r in rows:
        c[r["doc_id"]] += 1
    return dict(c)
