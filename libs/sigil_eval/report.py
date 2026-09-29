"""One command produces the full comparison report (Phase 8 acceptance).

    python -m sigil_eval.report --qrels golden/qrels.txt \
        --run sigil=runs/sigil.json --run hybrid=runs/hybrid.json --run bm25=runs/bm25.json \
        --metrics runs/sigil_system_metrics.json --incumbent runs/incumbent_metrics.json \
        --kind full_retrain --out report.json

A run file is ``{"qid": ["doc", ...]}``. ``--metrics`` carries what cannot be computed from
a run alone (valid_id_rate, ECE, channel shares, latency, cost, held-out recall) and is
merged over the ``sigil`` run's retrieval metrics before gating. Exit status 1 means BLOCK.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sigil_eval import gates
from sigil_eval.golden_sets import load_qrels
from sigil_eval.metrics import evaluate_run


def build(qrels: dict, runs: dict[str, dict], extra: dict, incumbent: dict | None, kind: str, thresholds: dict | None):
    systems = {name: evaluate_run(run, qrels) for name, run in runs.items()}
    candidate = {**systems.get("sigil", {}), **extra}
    verdict = gates.evaluate(candidate, incumbent, kind=kind, thresholds=thresholds, baseline=systems.get("bm25"))
    report = {
        "systems": systems,
        "gates": {"passed": verdict.passed, "results": [r.__dict__ for r in verdict.results]},
    }
    if "sigil" in systems and "hybrid" in systems:
        report["decision_baseline"] = {
            "sigil_ndcg@10": systems["sigil"]["ndcg@10"],
            "hybrid_ndcg@10": systems["hybrid"]["ndcg@10"],
            "within_3_points": gates.hybrid_decision(systems["sigil"]["ndcg@10"], systems["hybrid"]["ndcg@10"]),
        }
    if "sigil" in systems and "doc2query_bm25" in systems:
        # §25.2: the cheap use of the same expensive artefact. Matching it means the
        # generative model is not earning its cost.
        report["doc2query_bm25_gap_ndcg@10"] = systems["sigil"]["ndcg@10"] - systems["doc2query_bm25"]["ndcg@10"]
    return report, verdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--run", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--metrics", help="system metrics JSON for the candidate bundle")
    ap.add_argument("--incumbent", help="metrics JSON of the bundle in production")
    ap.add_argument("--kind", default=gates.FULL, choices=[gates.FULL, gates.ADAPTER])
    ap.add_argument("--config", default=str(gates.DEFAULT_THRESHOLDS))
    ap.add_argument("--out", default="report.json")
    a = ap.parse_args(argv)

    def load(p):
        return json.loads(Path(p).read_text()) if p else None

    runs = {name: load(path) for name, path in (r.split("=", 1) for r in a.run)}
    report, verdict = build(load_qrels(a.qrels), runs, load(a.metrics) or {}, load(a.incumbent), a.kind,
                            gates.load_thresholds(a.config))
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(verdict.summary())
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
