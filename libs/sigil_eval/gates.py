"""Release gates. The single source of truth.

CI, the training pipeline, and the deployment workflow all import ``evaluate``. Gates
defined in three places are gates enforced in zero (§31). Thresholds live in
``configs/gates/release_gates.yaml``; which metric each gate reads and which way it cuts
lives here.

A gate whose metric was not measured fails. A release that did not measure something it
is gated on has not passed that gate.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_THRESHOLDS = Path(__file__).resolve().parents[2] / "configs" / "gates" / "release_gates.yaml"

FULL, ADAPTER = "full_retrain", "adapter_refresh"


@dataclass(frozen=True)
class Gate:
    name: str
    metric: str
    threshold_key: str
    # (candidate value, threshold, incumbent value or None) -> passed
    check: Callable[[float, float, float | None], bool]
    kinds: frozenset[str] = frozenset({FULL, ADAPTER})
    relative: bool = False  # needs an incumbent value to evaluate


def _abs(op) -> Callable:
    return lambda v, t, _inc: op(v, t)


def _drop(v, t, inc):  # candidate may be at most t below incumbent
    return v >= inc - t


def _ratio(v, t, inc):  # candidate may be at most (1+t) times incumbent
    return v <= inc * (1 + t)


GATES: tuple[Gate, ...] = (
    Gate("valid_id_rate", "valid_id_rate", "valid_id_rate_exact", _abs(operator.eq)),
    Gate("ndcg@10", "ndcg@10", "ndcg_at_10_max_drop", _drop, relative=True),
    Gate("recall@10", "recall@10", "recall_at_10_max_drop_points",
         lambda v, t, inc: v >= inc - t / 100, relative=True),
    Gate("mrr@10", "mrr@10", "mrr_at_10_max_drop", _drop, relative=True),
    Gate("held_out_doc_recall@10", "held_out_doc_recall@10", "held_out_doc_recall_at_10_min", _abs(operator.ge)),
    Gate("old_doc_recall_regression", "old_doc_recall_regression_points",
         "old_doc_recall_regression_max_points", _abs(operator.le), kinds=frozenset({ADAPTER})),
    Gate("cold_doc_recall", "cold_doc_recall_ratio", "cold_doc_recall_min_ratio_of_steady_state",
         _abs(operator.ge), kinds=frozenset({ADAPTER})),
    Gate("ece", "ece", "ece_max", _abs(operator.le)),
    Gate("stale_id_rate", "stale_id_rate", "stale_id_rate_max", _abs(operator.le)),
    Gate("escape_rate", "escape_rate", "escape_rate_max", _abs(operator.le)),
    Gate("channel_c_query_share", "channel_c_query_share", "channel_c_query_share_max", _abs(operator.le)),
    Gate("channel_a_result_share", "channel_a_result_share", "channel_a_result_share_min", _abs(operator.ge)),
    Gate("p95_latency", "p95_latency_ms", "p95_latency_max_increase_ratio", _ratio, relative=True),
    Gate("cost_per_1k", "cost_per_1k_queries", "cost_per_1k_queries_max_increase_ratio", _ratio, relative=True),
    Gate("cross_tenant_leaks", "cross_tenant_leaks", "cross_tenant_leaks_max", _abs(operator.le)),
)


# What an offline training run can measure by itself. Channel shares, latency, cost, and the
# tenant-leak suite need shadow traffic and the security run; the release decision (§29)
# evaluates every gate, and an unmeasured one fails there.
OFFLINE = frozenset({"valid_id_rate", "ndcg@10", "recall@10", "mrr@10", "held_out_doc_recall@10", "ece",
                     "stale_id_rate", "escape_rate", "old_doc_recall_regression", "cold_doc_recall"})


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float | None
    threshold: float
    incumbent: float | None
    note: str = ""


@dataclass(frozen=True)
class Verdict:
    passed: bool
    results: tuple[GateResult, ...]

    @property
    def failures(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        lines = [f"{'PASS' if self.passed else 'BLOCK'}: {len(self.failures)} of {len(self.results)} gates failed"]
        for r in self.results:
            inc = "" if r.incumbent is None else f" (incumbent {r.incumbent:.4g})"
            val = "not measured" if r.value is None else f"{r.value:.4g}"
            lines.append(f"  [{'ok' if r.passed else 'FAIL'}] {r.name}: {val} vs {r.threshold:g}{inc} {r.note}".rstrip())
        return "\n".join(lines)


def load_thresholds(path: str | Path = DEFAULT_THRESHOLDS) -> dict[str, float]:
    return yaml.safe_load(Path(path).read_text())["gates"]


def evaluate(
    candidate: Mapping[str, float],
    incumbent: Mapping[str, float] | None = None,
    *,
    kind: str = FULL,
    thresholds: Mapping[str, float] | None = None,
    only: frozenset[str] | None = None,
) -> Verdict:
    """Apply every gate that applies to this release kind. First release (no incumbent):
    relative gates are skipped and say so; absolute gates still apply."""
    th = thresholds or load_thresholds()
    results = []
    for g in GATES:
        if kind not in g.kinds or (only is not None and g.name not in only):
            continue
        t = float(th[g.threshold_key])
        v = candidate.get(g.metric)
        inc = None if incumbent is None else incumbent.get(g.metric)
        if v is None:
            results.append(GateResult(g.name, False, None, t, inc, "metric not measured"))
        elif g.relative and inc is None:
            results.append(GateResult(g.name, True, v, t, None, "no incumbent; skipped"))
        else:
            results.append(GateResult(g.name, bool(g.check(float(v), t, inc)), float(v), t, inc))
    return Verdict(all(r.passed for r in results), tuple(results))


def hybrid_decision(sigil_ndcg: float, hybrid_ndcg: float, max_gap: float = 0.03) -> bool:
    """§25.2 decision baseline: SIGIL must be within 3 points of NDCG@10 of the hybrid to
    justify itself at all. Not a release gate; a Phase 0 / E0 stop condition."""
    return sigil_ndcg >= hybrid_ndcg - max_gap
