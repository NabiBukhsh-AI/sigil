"""Per-request trace (§28.1) and the closed-set miss diagnosis (§28.4).

The trace is a plain dict of spans so it can be logged, returned from /v1/debug/explain,
or exported to OpenTelemetry by the service layer without libs depending on OTel.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum


class MissReason(StrEnum):
    """Why did I not get this document? Always exactly one of these (§28.4).

    Making the set explicit is what turns generative retrieval from a black box into an
    operable system.
    """

    NOT_IN_TRIE = "not_in_trie_snapshot"
    PRUNED = "pruned_at_level"  # detail carries the level
    FILTERED_TOMBSTONE = "filtered_tombstone"
    FILTERED_ACL = "filtered_acl"
    COLD_START_OUTRANKED = "cold_start_outranked"
    RERANKED_BELOW_K = "reranked_below_k"
    LOW_RELEVANCE = "genuinely_low_relevance"
    RETRIEVED = "retrieved"


@dataclass
class Trace:
    trace_id: str
    attrs: dict = field(default_factory=dict)
    spans: dict[str, dict] = field(default_factory=dict)

    @contextmanager
    def span(self, name: str, **attrs):
        record = dict(attrs)
        self.spans[name] = record
        t0 = time.perf_counter()
        try:
            yield record
        finally:
            record["ms"] = round((time.perf_counter() - t0) * 1000, 3)

    def ms(self, name: str) -> float:
        return self.spans.get(name, {}).get("ms", 0.0)

    def to_dict(self) -> dict:
        return {"trace_id": self.trace_id, **self.attrs, "spans": self.spans}


# §28.2 metric names. Services register these with prometheus_client.
METRICS = (
    "sigil_retrieve_latency_ms",
    "sigil_valid_id_rate",
    "sigil_stale_id_rate",
    "sigil_trie_registry_skew_total",
    "sigil_confidence_histogram",
    "sigil_prefix_entropy",
    "sigil_channel_share",
    "sigil_results_from_channel",
    "sigil_fallback_trigger_total",
    "sigil_cold_start_doc_share",
    "sigil_hot_set_size_ratio",
    "sigil_escape_rate",
    "sigil_adapter_age_days",
    "sigil_trie_snapshot_age_seconds",
    "sigil_bundle_info",
    "sigil_duplicate_rate_at_k",
    "sigil_distinct_l1_codes_at_k",
    "sigil_rerank_generative_kendall_tau",
    "sigil_admission_rejections_total",
)
