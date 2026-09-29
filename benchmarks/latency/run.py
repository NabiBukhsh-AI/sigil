"""Per-stage latency through the full in-process stack. §22.1, Experiment 10.

    python benchmarks/latency/run.py --docs 1600 --queries 200

Reports p50/p95/p99 for the stages the gateway traces (decode, resolve, rerank, lexical) and
the total. The generative channel here is the quantizer router, so "decode" measures beam
search, trie masking, and fan-out overhead without model FLOPs: the part of decode cost this
codebase owns. Add the encoder and decoder forward times from a GPU profile to get the
§22.1 row totals.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from scripts.local_stack import build, query_for, synthetic_corpus


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 3)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=int, default=1600)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    from sigil_decoding import DecodeConfig

    docs = synthetic_corpus(16, a.docs // 16)
    with tempfile.TemporaryDirectory() as tmp:
        s = build(docs, Path(tmp), decode=DecodeConfig(beam=a.beam))
        try:
            for t, i, _ in docs[:20]:  # warm up
                s.ask(query_for(t, i))
            stages: dict[str, list[float]] = {}
            for t, i, _ in docs[: a.queries]:
                r = s.ask(query_for(t, i))
                for k, v in r["latency_ms"].items():
                    stages.setdefault(k, []).append(v)
        finally:
            s.close()
    out = {"docs": len(docs), "queries": a.queries, "beam": a.beam,
           "stages_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99)} for k, v in stages.items()}}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
