"""Admission control and decode orchestration for one GPU replica. §20.2 S2, §22.4.

One decode stream per replica behind an admission queue with a hard depth and a hard beam
cap. Requests past the budget get 429 with Retry-After instead of degrading everyone's
tail latency.

ponytail: requests decode one at a time on the stream; the admission queue bounds wait.
§22.3 marks cross-request continuous batching as mandatory for throughput. It fits here
unusually well because every request decodes exactly 4 steps: run the beams of queued
requests in lockstep, one concatenated decoder call per level. Add it when load tests
show the stream saturating.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

from sigil_core.errors import CapExceeded
from sigil_core.ids import SemanticId
from sigil_decoding import DecodeConfig, Scorer, decode
from sigil_decoding.confidence import Calibrator

from services.generative_retrieval.snapshot_manager import SnapshotManager


@dataclass
class EngineLimits:
    max_beam: int = 128
    max_candidate_cap: int = 400
    max_queue: int = 64


class Engine:
    def __init__(self, scorer: Scorer, snapshots: SnapshotManager, calibrator: Calibrator = Calibrator(),
                 limits: EngineLimits = EngineLimits()):
        self.scorer, self.snapshots, self.cal, self.limits = scorer, snapshots, calibrator, limits
        self._stream = threading.Lock()  # the one decode stream
        self._queued = 0
        self._qlock = threading.Lock()
        self.rejections = 0

    def retrieve(self, query: str, cfg: DecodeConfig, *, explain: bool = False, probe: str | None = None) -> dict:
        if cfg.beam > self.limits.max_beam:
            self.rejections += 1
            raise CapExceeded(f"beam {cfg.beam} exceeds cap {self.limits.max_beam}")
        cfg = cfg.clamp(self.limits.max_beam, self.limits.max_candidate_cap)
        with self._qlock:
            if self._queued >= self.limits.max_queue:
                self.rejections += 1
                raise CapExceeded("admission queue full")
            self._queued += 1
        try:
            with self._stream, self.snapshots.acquire() as trie:
                t0 = time.perf_counter()
                state = self.scorer.encode(query)
                t1 = time.perf_counter()
                r = decode(self.scorer, state, trie, cfg, self.cal, encoded=True)
                t2 = time.perf_counter()
                snapshot = trie.version
                probe_in = SemanticId.parse(probe) in trie if probe else None
        finally:
            with self._qlock:
                self._queued -= 1
        out = {
            "candidates": [{"semantic_id": str(c.sid), "score": c.score, "prefix_score": c.prefix_score,
                            "prefix_rank": c.prefix_rank} for c in r.candidates],
            "confidence": r.confidence, "margin_1": r.margin_1, "s_top": r.s_top,
            "beam_width": r.beam.beam_width, "trie_snapshot": snapshot,
            "latency_ms": {"encode": (t1 - t0) * 1e3, "decode": (t2 - t1) * 1e3},
            "probe_in_trie": probe_in,
        }
        if explain:
            out["beam_trace"] = [lv.__dict__ for lv in r.beam.levels]
            out["survivors"] = [[".".join(map(str, p)) for p in s] for s in r.beam.survivors]
        return out

    def retrieve_widened(self, query: str, cfg: DecodeConfig, **kw) -> dict:
        """§18.2: the one retry on low confidence. Wider beam, multi-start on."""
        return self.retrieve(query, replace(cfg, beam=cfg.widened_beam, multi_start=True, adaptive_beam=False), **kw)
