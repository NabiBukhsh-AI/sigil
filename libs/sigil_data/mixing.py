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


def mix(rows: Sequence[dict], ratios: Mapping[str, float], total: int, seed: int = 0) -> list[dict]:
    """Sample ``total`` rows matching family ratios. A family short of its quota passes the
    remainder on to families with room, rather than being oversampled with repeats (real
    logs are usually short early on)."""
    rng = np.random.default_rng(seed)
    fams: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        fams[FAMILY_OF_SOURCE[r["source"]]].append(i)
    quota = {f: int(round(ratios.get(f, 0.0) * total)) for f in fams}
    picked: list[int] = []
    spare = 0
    for f in sorted(fams, key=lambda f: (len(fams[f]) - quota[f], f)):  # shortest-handed first
        q = quota[f] + spare
        take = min(q, len(fams[f]))
        spare = q - take
        picked += list(rng.choice(fams[f], take, replace=False))
    picked = sorted(picked)[:total]
    rng.shuffle(picked)
    return [rows[i] for i in picked]


def coverage(rows: Sequence[dict]) -> dict[str, int]:
    """Examples per unit, for the ``min_queries_per_doc`` floor check."""
    c: dict[str, int] = defaultdict(int)
    for r in rows:
        c[r["doc_id"]] += 1
    return dict(c)
