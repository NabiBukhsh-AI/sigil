"""Trie scale sweep. Phase 3 acceptance, §1.2, §23.1, Experiment 9 (index side).

    python benchmarks/scale/run.py --corpus 10k 100k 1m

For each size: build time, load-and-verify time, file bytes, bytes per document, and the
per-step decode cost at beam 64 (mask + child lookup). Identifiers are drawn from a clumpy
distribution over 256^4 codes so prefix sharing resembles a fitted quantizer, not uniform noise.

Acceptance targets: a 10M-document trie builds in under 2 minutes and loads in under 3 s;
footprint 120 to 200 MB at 10M (roughly two orders of magnitude under a flat fp16 768-d vector
index, ~15 GB). Sizes here extrapolate linearly in N.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from sigil_core.ids import SemanticId
from sigil_trie import TrieSnapshot, write

SIZES = {"10k": 10_000, "100k": 100_000, "1m": 1_000_000, "10m": 10_000_000}


def synthetic_ids(n: int, seed: int = 0) -> list[SemanticId]:
    """Heavy-tailed coarse levels, near-uniform fine levels, like a balanced RQ fit."""
    rng = np.random.default_rng(seed)
    l1 = rng.zipf(1.3, n) % 256
    l2 = (l1 * 31 + rng.zipf(1.5, n)) % 256
    l3 = rng.integers(0, 256, n)
    l4 = rng.integers(0, 256, n)
    codes = np.stack([l1, l2, l3, l4], 1)
    order = np.lexsort(codes.T[::-1])
    codes = codes[order]
    same = np.r_[False, (codes[1:] == codes[:-1]).all(1)]
    ordinal = np.zeros(n, dtype=np.int64)
    for i in np.flatnonzero(same):  # consecutive duplicates of a prefix take the next ordinal
        ordinal[i] = ordinal[i - 1] + 1
    return [SemanticId.from_ordinal(tuple(int(c) for c in row), int(o)) for row, o in zip(codes, ordinal, strict=True)]


def step_cost(trie: TrieSnapshot, beam: int = 64, reps: int = 200, seed: int = 0) -> dict:
    """Per-step mask + child cost at the deepest level, where the beam is widest."""
    rng = np.random.default_rng(seed)
    depth = 3
    n_nodes = len(trie._a[f"child_off_{depth}"]) - 1
    times_mask, times_child = [], []
    for _ in range(reps):
        nodes = rng.integers(0, n_nodes, beam)
        t0 = time.perf_counter()
        mask = trie.children_mask(nodes, depth)
        t1 = time.perf_counter()
        rows, codes = np.nonzero(mask)
        trie.child(nodes[rows], depth, codes)
        t2 = time.perf_counter()
        times_mask.append((t1 - t0) * 1e3)
        times_child.append((t2 - t1) * 1e3)
    q = lambda xs, p: float(np.percentile(xs, p))  # noqa: E731
    return {"mask_p50_ms": q(times_mask, 50), "mask_p95_ms": q(times_mask, 95),
            "child_p50_ms": q(times_child, 50), "child_p95_ms": q(times_child, 95)}


def run(n: int, workdir: Path) -> dict:
    t0 = time.perf_counter()
    ids = synthetic_ids(n)
    t_gen = time.perf_counter() - t0
    t0 = time.perf_counter()
    path, sha = write(ids, workdir, id_schema="ids_v1", corpus_snapshot="cs_bench")
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    trie = TrieSnapshot(path, sha)
    t_load = time.perf_counter() - t0
    try:
        steps = step_cost(trie)
        escaped = sum(s.escaped for s in ids)
    finally:
        trie.close()
    size = Path(path).stat().st_size
    return {"n": n, "gen_s": round(t_gen, 2), "build_s": round(t_build, 2), "load_verify_s": round(t_load, 3),
            "bytes": size, "bytes_per_doc": round(size / n, 2), "mb": round(size / 2**20, 2),
            "extrapolated_10m_mb": round(size / n * 10_000_000 / 2**20, 1), "escape_rate": escaped / n, **steps}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", default=["10k", "100k"], choices=sorted(SIZES))
    ap.add_argument("--out", help="write results JSON here")
    a = ap.parse_args(argv)
    results = []
    for name in a.corpus:
        with tempfile.TemporaryDirectory() as tmp:
            results.append({"corpus": name, **run(SIZES[name], Path(tmp))})
            print(json.dumps(results[-1]))
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
