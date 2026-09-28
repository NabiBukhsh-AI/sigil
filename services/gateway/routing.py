"""Query classification, channel policy, and the GPU circuit breaker. §14.2, §14.3.

Exact-identifier, quoted, and symbol-heavy queries go to the lexical channel by routing,
not by failure: BM25 wins these outright and generative retrieval loses them badly (§1.8).
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque

_QUOTED = re.compile(r'"[^"]{2,}"')
_SYMBOL = re.compile(
    r"[A-Za-z_][\w]*(::|->|\.)[A-Za-z_]\w*\(?"  # code symbols: Foo::bar, obj.method(
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-"  # uuids
    r"|\b[A-Z]{2,}[-_]?\d{2,}\b"  # ticket / sku ids: ABC-1234
    r"|\b\w*_\w+\b"  # snake_case identifiers
    r"|\b\d{5,}\b"  # long numbers
)

SEMANTIC, LEXICAL = "semantic", "lexical"

# Channel C trigger reason codes, logged on every invocation (§14.2).
LOW_CONFIDENCE = "low_confidence"
LOW_MARGIN = "low_margin"
VERIFY_DROPPED_MAJORITY = "verification_dropped_majority"
IDENTIFIER_QUERY = "identifier_like_query"
TOO_FEW_DISTINCT = "fewer_than_k_distinct"
GPU_BREAKER_OPEN = "gpu_breaker_open"
GENERATIVE_ERROR = "generative_error"
LOW_RELEVANCE = "low_relevance"


def classify(query: str) -> str:
    return LEXICAL if _QUOTED.search(query) or _SYMBOL.search(query) else SEMANTIC


class ChannelLedger:
    """Rolling 24 h share of queries that invoked channel C. Above 15% is a release
    blocker; above 25% pages on-call. Tracked, not enforced per request: refusing the
    fallback would trade creep for outages, and the fix belongs in channel A."""

    def __init__(self, window_s: float = 86_400, alert: float = 0.15, page: float = 0.25):
        self.window, self.alert, self.page = window_s, alert, page
        self._q: deque[tuple[float, bool]] = deque()
        self._lock = threading.Lock()

    def record(self, invoked_c: bool, t: float | None = None) -> None:
        t = time.time() if t is None else t
        with self._lock:
            self._q.append((t, invoked_c))
            while self._q and self._q[0][0] < t - self.window:
                self._q.popleft()

    def share(self) -> float:
        with self._lock:
            return sum(c for _, c in self._q) / len(self._q) if self._q else 0.0

    def status(self) -> str:
        s = self.share()
        return "page" if s > self.page else "alert" if s > self.alert else "ok"


class CircuitBreaker:
    """§24 F13. Consecutive generative failures open the breaker; traffic goes lexical with
    ``degraded_mode=true`` until a half-open probe succeeds."""

    def __init__(self, failures: int = 5, cooldown_s: float = 10.0):
        self.threshold, self.cooldown = failures, cooldown_s
        self.failures, self.opened_at = 0, None
        self._lock = threading.Lock()

    @property
    def open(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return False
            if time.monotonic() - self.opened_at >= self.cooldown:
                self.opened_at = None  # half-open: let one through
                self.failures = self.threshold - 1
                return False
            return True

    def success(self) -> None:
        with self._lock:
            self.failures, self.opened_at = 0, None

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = time.monotonic()
