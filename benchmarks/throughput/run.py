"""Throughput and admission control under concurrency. §22.4, §33 load row.

    python benchmarks/throughput/run.py --threads 1 4 16 --seconds 10

Drives the generative engine from N threads. What to look for: throughput plateaus at the
single decode stream's rate, and past the queue budget requests are rejected with 429
(CapExceeded) instead of the p99 of accepted requests growing without bound.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from sigil_core.errors import CapExceeded

from scripts.local_stack import build, query_for, synthetic_corpus
from services.generative_retrieval.engine import EngineLimits


def drive(engine, cfg, queries, threads: int, seconds: float) -> dict:
    lat, rejected, stop = [], [0], time.perf_counter() + seconds
    lock = threading.Lock()

    def worker(k):
        n = k
        while time.perf_counter() < stop:
            q = queries[n % len(queries)]
            n += threads
            t0 = time.perf_counter()
            try:
                engine.retrieve(q, cfg)
                with lock:
                    lat.append((time.perf_counter() - t0) * 1e3)
            except CapExceeded:
                with lock:
                    rejected[0] += 1
                time.sleep(0.005)  # a well-behaved client honours Retry-After instead of spinning

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return {"threads": threads, "qps": round(len(lat) / seconds, 1), "rejected": rejected[0],
            "p50_ms": round(float(np.percentile(lat, 50)), 2) if lat else None,
            "p99_ms": round(float(np.percentile(lat, 99)), 2) if lat else None}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--max-queue", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    docs = synthetic_corpus(16, 50)
    with tempfile.TemporaryDirectory() as tmp:
        s = build(docs, Path(tmp))
        s.engine.limits = EngineLimits(max_queue=a.max_queue)
        try:
            queries = [query_for(t, i) for t, i, _ in docs]
            rows = [drive(s.engine, s.gw.cfg.decode, queries, n, a.seconds) for n in a.threads]
        finally:
            s.close()
    for r in rows:
        print(json.dumps(r))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
