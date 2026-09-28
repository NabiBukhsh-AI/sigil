"""S1 Retrieval Gateway. §3.2, §18.1, §18.2, §20.2.

Per request: authenticate, validate, clamp, normalize, classify; dispatch the generative and
hot lexical channels in parallel under per-channel deadlines; widen once on low confidence;
invoke full lexical only on a logged trigger; resolve and tier-1 verify; collapse duplicates;
apply the diversity quota; rerank; blend; hard-check the final k against the registry; and
stamp bundle, trace, and per-document channel attribution on the response.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from sigil_core.ids import SemanticId
from sigil_core.telemetry import MissReason, Trace
from sigil_decoding import DecodeConfig
from sigil_model.tokenization import normalize
from sigil_registry_client.bloom import Bloom
from sigil_registry_client.records import DocState, Principal

from services.gateway import merge, routing
from services.gateway.schemas import ExplainRequest, RetrieveRequest
from services.gateway.verification import Cand, tier1

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    M_LAT = Histogram("sigil_retrieve_latency_ms", "latency by stage", ["stage"],
                      buckets=(1, 2, 5, 10, 20, 40, 60, 100, 150, 250, 500, 1000))
    M_FALLBACK = Counter("sigil_fallback_trigger_total", "channel C triggers", ["reason"])
    M_SKEW = Counter("sigil_trie_registry_skew_total", "trie ids missing from registry")
    M_RESULTS = Counter("sigil_results_from_channel", "returned docs by channel", ["channel"])
    M_C_SHARE = Gauge("sigil_channel_share", "rolling query share", ["channel"])
    M_DISTINCT_L1 = Histogram("sigil_distinct_l1_codes_at_k", "distinct level-1 codes in the final k")
    M_CONF = Histogram("sigil_confidence_histogram", "calibrated generative confidence",
                       buckets=tuple(i / 10 for i in range(11)))
except ImportError:  # metrics are optional for library-style use in tests
    M_LAT = M_FALLBACK = M_SKEW = M_RESULTS = M_C_SHARE = M_DISTINCT_L1 = M_CONF = None


def _obs(metric, *labels, value: float | None = None, inc: float | None = None) -> None:
    if metric is None:
        return
    m = metric.labels(*labels) if labels else metric
    if value is not None:
        m.observe(value) if hasattr(m, "observe") else m.set(value)
    if inc is not None:
        m.inc(inc)


class NoChannel(Exception):
    """No retrieval channel could serve the request. HTTP 503. Never serve unverified."""


@dataclass
class Backends:
    generative: Callable[..., dict]  # (query, cfg: DecodeConfig, widen, explain, probe) -> GRS response
    hot_lexical: Callable[[str, int], list[tuple[str, float]]]
    full_lexical: Callable[[str, int], list[tuple[str, float]]]
    registry: object
    rerank: Callable[[str, list[str]], np.ndarray] | None


@dataclass
class GatewayConfig:
    bundle_id: str
    id_schema: str
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    weights: merge.Weights = field(default_factory=merge.Weights)
    tau_conf: float = 0.30
    tau_margin: float = 0.50
    tau_relevant: float = 0.35
    max_beam: int = 128
    max_candidate_cap: int = 400
    hot_n: int = 25
    full_n: int = 50
    deadlines_ms: dict = field(default_factory=lambda: {"generative": 60, "lexical": 25, "rerank": 40})
    bundle_epoch_range: tuple[int, int] = (0, 2**62)
    cache_entries: int = 100_000


class LRU:
    def __init__(self, n: int):
        self.n, self.d, self.lock = n, OrderedDict(), threading.Lock()

    def get(self, k):
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]
        return None

    def put(self, k, v) -> None:
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.n:
                self.d.popitem(last=False)


class Gateway:
    def __init__(self, cfg: GatewayConfig, backends: Backends):
        self.cfg, self.b = cfg, backends
        self.ledger = routing.ChannelLedger()
        self.breaker = routing.CircuitBreaker()
        self.cache = LRU(cfg.cache_entries)
        self.tombstones = Bloom(capacity=1000)
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.audit: list[dict] = []

    def refresh_tombstones(self) -> None:
        """Run every 2 s in production (§12.3)."""
        self.tombstones = self.b.registry.tombstone_bloom()

    # -- channels ---------------------------------------------------------------------------

    def _generative(self, query: str, cfg: DecodeConfig, trace: Trace, reasons: list[str], **kw) -> dict | None:
        if self.breaker.open:
            reasons.append(routing.GPU_BREAKER_OPEN)
            return None
        key = hashlib.sha256(f"{query}|{self.cfg.bundle_id}|{self.b.registry.epoch}|{cfg.beam}|{cfg.candidate_cap}|"
                             f"{kw.get('widen', False)}".encode()).hexdigest()
        if not kw.get("explain") and (hit := self.cache.get(key)) is not None:
            trace.attrs["cache_hit"] = True
            return hit
        fut = self.pool.submit(self.b.generative, query, cfg, **kw)
        try:
            out = fut.result(timeout=self.cfg.deadlines_ms["generative"] / 1000 * (4 if kw.get("widen") else 1))
        except FutureTimeout:
            self.breaker.failure()
            reasons.append(routing.GENERATIVE_ERROR)
            trace.attrs["generative_timeout"] = True
            return None
        except Exception as e:  # noqa: BLE001  any GRS failure degrades to lexical
            self.breaker.failure()
            reasons.append(routing.GENERATIVE_ERROR)
            trace.attrs["generative_error"] = repr(e)[:200]
            return None
        self.breaker.success()
        if not kw.get("explain"):
            self.cache.put(key, out)
        return out

    def _lexical(self, fn, query: str, n: int, channel: str) -> list[Cand]:
        return [Cand(SemanticId.parse(k), channel, bm25=s) for k, s in fn(query, n)]

    # -- the request ------------------------------------------------------------------------

    def retrieve(self, req: RetrieveRequest, principal: Principal, *, explain: bool = False, probe: str | None = None):
        t_start = time.perf_counter()
        trace = Trace(uuid.uuid4().hex[:16].upper(), {"bundle_id": self.cfg.bundle_id, "tenant": req.tenant_id,
                                                         "k": req.k})
        o = req.options
        cfg = replace(self.cfg.decode, beam=o.beam_width, candidate_cap=o.candidate_cap,
                      quota_l1=o.diversity_quota).clamp(self.cfg.max_beam, self.cfg.max_candidate_cap)
        reasons: list[str] = []
        degraded = False
        query = normalize(req.query)
        with trace.span("gateway.route") as s:
            route = routing.classify(query)
            s["route_decision"] = route
        trace.attrs["query_hash"] = hashlib.sha256(query.encode()).hexdigest()[:16]

        # §18.1 E8: a registry older than the bundle means a version mismatch.
        registry_epoch = self.b.registry.epoch
        if registry_epoch < self.cfg.bundle_epoch_range[0]:
            degraded = True
            trace.attrs["version_mismatch"] = {"registry_epoch": registry_epoch, "bundle": self.cfg.bundle_epoch_range}

        gen, cands = None, []
        if route == routing.SEMANTIC:
            hot_f = self.pool.submit(self._lexical, self.b.hot_lexical, query, self.cfg.hot_n, "hot_lexical")
            with trace.span("generative.decode") as s:
                gen = self._generative(query, cfg, trace, reasons, explain=explain, probe=probe)
                if gen is not None and (gen["confidence"] < self.cfg.tau_conf or gen["margin_1"] < self.cfg.tau_margin):
                    # §18.2: widen once, then merge with lexical if still unsure.
                    s["widened"] = True
                    wider = self._generative(query, cfg, trace, reasons, widen=True, explain=explain, probe=probe)
                    gen = wider if wider is not None and wider["confidence"] >= gen["confidence"] else gen
                    if gen["confidence"] < self.cfg.tau_conf:
                        reasons.append(routing.LOW_CONFIDENCE)
                    elif gen["margin_1"] < self.cfg.tau_margin:
                        reasons.append(routing.LOW_MARGIN)
                if gen is not None:
                    s.update(beam=gen["beam_width"], candidates=len(gen["candidates"]), confidence=gen["confidence"],
                             margin_1=gen["margin_1"], trie_snapshot=gen.get("trie_snapshot"))
                    _obs(M_CONF, value=gen["confidence"])
                    cands += [Cand(SemanticId.parse(c["semantic_id"]), "generative", gen=c["score"])
                              for c in gen["candidates"]]
                else:
                    degraded = True
            with trace.span("lexical.hot") as s:
                try:
                    hot = hot_f.result(timeout=self.cfg.deadlines_ms["lexical"] / 1000 * 4)
                except Exception:  # noqa: BLE001  the hot channel is best effort
                    hot, degraded = [], True
                s["hits"] = len(hot)
                cands += hot
        else:
            reasons.append(routing.IDENTIFIER_QUERY)

        def fetch_full() -> list[Cand]:
            with trace.span("lexical.full") as s:
                out = self._lexical(self.b.full_lexical, query, self.cfg.full_n, "full_lexical")
                s["hits"] = len(out)
            return out

        full_invoked = bool(reasons) and o.allow_fallback
        if full_invoked:
            cands += fetch_full()
        elif gen is None and route == routing.SEMANTIC and not cands:
            raise NoChannel("generative unavailable and fallback disallowed")

        with trace.span("registry.resolve") as s:
            resolved = self.b.registry.resolve([c.sid for c in cands])
            s.update(ids_in=len(cands), hits=len(resolved))
        with trace.span("verify.tier1") as s:
            v = tier1(cands, resolved, principal, self.cfg.id_schema, self.tombstones, req.filters, self.audit)
            s.update(dropped=dict(v.dropped), aliases_applied=v.aliases)
            _obs(M_SKEW, inc=v.skew)
        n_gen = sum(c.channel == "generative" for c in cands)
        gen_dropped = n_gen - sum(c.channel == "generative" for c in v.kept)
        if not full_invoked and o.allow_fallback and (
            (n_gen and gen_dropped > n_gen / 2) or len(v.kept) < req.k
        ):
            reasons.append(routing.VERIFY_DROPPED_MAJORITY if n_gen and gen_dropped > n_gen / 2 else routing.TOO_FEW_DISTINCT)
            extra = fetch_full()
            more = tier1(v.kept + extra, self.b.registry.resolve([c.sid for c in v.kept + extra]), principal,
                         self.cfg.id_schema, self.tombstones, req.filters, self.audit)
            v.kept, full_invoked = more.kept, True

        pool = merge.collapse_near_duplicates(merge.preliminary_order(v.kept))
        pool = merge.apply_quota(pool, min(len(pool), self.cfg.max_candidate_cap), o.diversity_quota)

        with trace.span("rerank") as s:
            if o.rerank and self.b.rerank is not None and pool:
                try:
                    texts = [f"{c.record.title or ''} {c.record.rerank_snippet or ''}".strip() for c in pool]
                    ce = self.b.rerank(query, texts)
                    for c, p in zip(pool, ce, strict=True):
                        c.ce = float(p)
                    s.update(pairs=len(pool), score_min=float(np.min(ce)), score_max=float(np.max(ce)))
                except Exception as e:  # noqa: BLE001  §24 F19: serve generative order, flag it
                    degraded = True
                    s["error"] = repr(e)[:200]
            elif o.rerank:
                degraded = True
        merge.blend(pool, self.cfg.weights)

        # §13.4: weak top result -> flag it and attach lexical results rather than hand weak
        # documents to an LLM that will treat them as authoritative.
        low_conf = bool(pool) and pool[0].ce is not None and pool[0].ce < self.cfg.tau_relevant
        if low_conf and not full_invoked and o.allow_fallback:
            reasons.append(routing.LOW_RELEVANCE)
            full_invoked = True
            extra = fetch_full()
            lex = tier1(extra, self.b.registry.resolve([c.sid for c in extra]), principal, self.cfg.id_schema,
                        self.tombstones, req.filters, self.audit)
            known = {c.record.doc_uid for c in pool}
            new = [c for c in lex.kept if c.record.doc_uid not in known]
            if new and all(c.ce is not None for c in pool):
                try:
                    ce = self.b.rerank(query, [f"{c.record.title or ''} {c.record.rerank_snippet or ''}".strip()
                                               for c in new])
                    for c, p in zip(new, ce, strict=True):
                        c.ce = float(p)
                except Exception:  # noqa: BLE001
                    degraded = True
            pool += new
            merge.blend(pool, self.cfg.weights)

        # Hard registry check of the returned k, never from cache alone (§13.1, §34).
        with trace.span("assemble") as s:
            final: list[Cand] = []
            fresh = self.b.registry.resolve([c.sid for c in pool[: req.k * 2]], fresh=True)
            for c in pool:
                if len(final) == req.k:
                    break
                hit = fresh.get(c.sid)
                if hit and hit.record.state != DocState.TOMBSTONED and hit.record.allows(principal):
                    final.append(c)
            mix = {ch: sum(c.channel == ch for c in final) for ch in merge.CHANNEL_ORDER}
            s.update(final_k=len(final), channel_mix=mix, degraded_mode=degraded,
                     diversity_l1_codes=len({c.sid.codes[0] for c in final}))

        self.ledger.record(full_invoked)
        for r in dict.fromkeys(reasons):
            _obs(M_FALLBACK, r, inc=1)
        for ch, n in mix.items():
            _obs(M_RESULTS, ch, inc=n)
        _obs(M_C_SHARE, "c", value=self.ledger.share())
        _obs(M_DISTINCT_L1, value=len({c.sid.codes[0] for c in final}))
        total_ms = (time.perf_counter() - t_start) * 1e3
        for stage in ("generative.decode", "lexical.hot", "registry.resolve", "rerank"):
            _obs(M_LAT, stage, value=trace.ms(stage))
        _obs(M_LAT, "total", value=total_ms)

        resp = {
            "trace_id": trace.trace_id,
            "bundle_id": self.cfg.bundle_id,
            "degraded_mode": degraded,
            "low_confidence": low_conf,
            "latency_ms": {"total": round(total_ms, 3), "decode": trace.ms("generative.decode"),
                           "resolve": trace.ms("registry.resolve"), "rerank": trace.ms("rerank"),
                           "lexical": trace.ms("lexical.hot") + trace.ms("lexical.full")},
            "channels": {
                "generative": {"candidates": n_gen, "confidence": gen["confidence"] if gen else None,
                               "margin_1": gen["margin_1"] if gen else None},
                "hot_lexical": {"candidates": sum(c.channel == "hot_lexical" for c in cands)},
                "full_lexical": {"invoked": full_invoked, "reasons": list(dict.fromkeys(reasons))},
            },
            "results": [{
                "doc_uid": c.record.doc_uid, "semantic_id": str(c.record.semantic_id), "title": c.record.title,
                "snippet": c.record.rerank_snippet, "channel": c.channel, "state": c.record.state.value,
                "scores": {"final": round(c.final, 6), "cross_encoder": c.ce, "generative": c.gen, "bm25": c.bm25},
                "content_uri": c.record.content_uri, "content_trust": c.record.content_trust,
                "metadata": c.record.metadata,
            } for c in final],
        }
        if explain or o.explain:
            resp["trace"] = trace.to_dict()
        return resp, gen, v, pool

    # -- §28.4 "why did I not get this document?" ----------------------------------------------

    def explain(self, req: ExplainRequest, principal: Principal) -> dict:
        gold = None
        if req.expected:
            if req.expected.count(".") >= 4:
                hit = self.b.registry.resolve([SemanticId.parse(req.expected)]).get(SemanticId.parse(req.expected))
                gold = hit.record if hit else None
            else:
                gold = self.b.registry.get(req.expected)
        rr = RetrieveRequest(query=req.query, k=req.k, tenant_id=req.tenant_id)
        resp, gen, v, pool = self.retrieve(rr, principal, explain=True,
                                           probe=str(gold.semantic_id) if gold else None)
        out = {"query": req.query, "beam_trace": (gen or {}).get("beam_trace"), "trace": resp["trace"],
               "verification": {"dropped": dict(v.dropped), "aliases": v.aliases}}
        if req.expected:
            out["oracle_check"] = diagnose(gold, gen, pool, resp, principal, req.k)
        return out


def diagnose(gold, gen: dict | None, pool: list[Cand], resp: dict, principal: Principal, k: int) -> dict:
    """Map a missing document to exactly one reason from the closed set."""
    if gold is None:
        return {"reason": MissReason.NOT_IN_TRIE, "detail": "unknown to the registry"}
    base = {"gold_doc": gold.doc_uid, "gold_id": str(gold.semantic_id)}
    if gold.state == DocState.TOMBSTONED:
        return {**base, "reason": MissReason.FILTERED_TOMBSTONE}
    if not gold.allows(principal):
        return {**base, "reason": MissReason.FILTERED_ACL}
    ranked = [c.record.doc_uid for c in pool]
    final = [r["doc_uid"] for r in resp["results"]]
    if gold.doc_uid in final:
        return {**base, "reason": MissReason.RETRIEVED, "final_rank": final.index(gold.doc_uid) + 1}
    if gen is not None and gen.get("probe_in_trie") is False:
        return {**base, "reason": MissReason.NOT_IN_TRIE, "detail": "absent from the active trie snapshot"}
    if gold.doc_uid in ranked:
        c = pool[ranked.index(gold.doc_uid)]
        if c.ce is not None and c.ce < 0.35:
            return {**base, "reason": MissReason.LOW_RELEVANCE, "cross_encoder": c.ce}
        reason = MissReason.COLD_START_OUTRANKED if gold.state == DocState.ACTIVE_COLD_START else MissReason.RERANKED_BELOW_K
        return {**base, "reason": reason, "final_rank": ranked.index(gold.doc_uid) + 1}
    survived = 0
    for level, prefixes in enumerate((gen or {}).get("survivors", []), start=1):
        if ".".join(map(str, gold.semantic_id.codes[:level])) in prefixes:
            survived = level
        else:
            break
    return {**base, "reason": MissReason.PRUNED, "survived_to_level": survived, "pruned_at_level": survived + 1}


# -- HTTP ---------------------------------------------------------------------------------------


def create_app(gw: Gateway, api_keys: dict[str, dict], rate_per_s: float = 200.0) -> FastAPI:
    app = FastAPI(title="sigil-gateway")
    buckets: dict[str, list[float]] = {}
    lock = threading.Lock()

    def principal(authorization: str = Header("")) -> Principal:
        key = authorization.removeprefix("Bearer ").strip()
        p = api_keys.get(key)
        if p is None:
            raise HTTPException(401, "unknown credentials")
        with lock:  # token bucket per tenant (§27)
            tokens, last = buckets.get(p["tenant"], [rate_per_s, time.monotonic()])
            now = time.monotonic()
            tokens = min(rate_per_s, tokens + (now - last) * rate_per_s)
            if tokens < 1:
                raise HTTPException(429, "rate limited", headers={"Retry-After": "1"})
            buckets[p["tenant"]] = [tokens - 1, now]
        return Principal(p["tenant"], frozenset(p.get("roles", [])), frozenset(p.get("scopes", [])))

    @app.post("/v1/retrieve")
    def retrieve(req: RetrieveRequest, p: Principal = Depends(principal)):
        if req.tenant_id != p.tenant_id:
            raise HTTPException(403, "tenant mismatch")
        if req.options.bundle_id and req.options.bundle_id != gw.cfg.bundle_id:
            raise HTTPException(409, f"bundle {req.options.bundle_id} is not active")
        try:
            resp, *_ = gw.retrieve(req, p)
        except NoChannel as e:
            return JSONResponse({"detail": str(e)}, status_code=503)
        return resp

    @app.post("/v1/debug/explain")
    def explain(req: ExplainRequest, p: Principal = Depends(principal)):
        if "operator" not in p.scopes:
            raise HTTPException(403, "explain requires operator scope")
        if req.tenant_id != p.tenant_id:
            raise HTTPException(403, "tenant mismatch")
        return gw.explain(req, p)

    @app.get("/v1/health")
    def health():
        return {"ok": True}

    @app.get("/v1/ready")
    def ready():
        return {"ready": True, "bundle_id": gw.cfg.bundle_id, "channel_c_share": gw.ledger.share(),
                "channel_c_status": gw.ledger.status(), "gpu_breaker_open": gw.breaker.open}

    @app.get("/v1/metrics")
    def metrics():
        if M_LAT is None:
            raise HTTPException(501, "prometheus_client not installed")
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def local_backends(engine, hot, full, registry, rerank=None) -> Backends:
    """Dev and test wiring: every channel in-process. ``hot``/``full`` are BM25 indexes
    keyed by semantic id string; ``engine`` is a generative_retrieval Engine."""

    def generative(query, dcfg, widen=False, explain=False, probe=None):
        fn = engine.retrieve_widened if widen else engine.retrieve
        return fn(query, dcfg, explain=explain, probe=probe)

    return Backends(generative, hot.search, full.search, registry, rerank)


def remote_backends(cfg: dict) -> Backends:
    """Production wiring: every channel is its own service (§20.1: per-request coupling)."""
    import httpx

    from sigil_registry_client.http import HttpRegistry

    ep = cfg["endpoints"]
    http = httpx.Client(timeout=1.0)

    def generative(query, dcfg: DecodeConfig, widen=False, explain=False, probe=None):
        r = http.post(f"{ep['generative']}/retrieve", json={"query": query, "beam": dcfg.beam,
                      "candidate_cap": dcfg.candidate_cap, "widen": widen, "explain": explain, "probe": probe})
        r.raise_for_status()
        return r.json()

    def lexical(url):
        def search(query, n):
            r = http.post(f"{url}/search", json={"query": query, "n": n})
            r.raise_for_status()
            return [(h["key"], h["score"]) for h in r.json()["hits"]]
        return search

    def rerank(query, texts):
        r = http.post(f"{ep['reranker']}/rerank", json={"query": query, "texts": texts})
        r.raise_for_status()
        return np.array(r.json()["scores"])

    return Backends(generative, lexical(ep["hot_lexical"]), lexical(ep["full_lexical"]),
                    HttpRegistry(ep["registry"]), rerank)


def from_env() -> FastAPI:
    from sigil_core.bundle import BundleManifest
    from sigil_core.config import load_yaml

    cfg = load_yaml(os.environ.get("SIGIL_CONFIG", "configs/serving/dev.yaml"))
    manifest = BundleManifest.load(os.environ["SIGIL_BUNDLE_MANIFEST"])
    v, lim = cfg["verification"], cfg["limits"]
    dec_yaml = load_yaml(cfg["decoding_profile"])
    gcfg = GatewayConfig(
        bundle_id=manifest.bundle_id, id_schema=manifest.id_schema,
        decode=DecodeConfig.from_yaml_dict(dec_yaml), weights=merge.Weights.from_cfg(cfg["blending"]),
        tau_conf=dec_yaml["confidence"]["tau_conf"], tau_margin=dec_yaml["confidence"]["tau_margin"],
        tau_relevant=v["tau_relevant"], max_beam=lim["max_beam_client"],
        max_candidate_cap=lim["max_candidate_cap_client"], deadlines_ms=cfg["deadlines_ms"],
        bundle_epoch_range=tuple(manifest.corpus_epoch_range),
    )
    gw = Gateway(gcfg, remote_backends(cfg))

    def tombstone_loop():
        while True:
            try:
                gw.refresh_tombstones()
            except Exception:  # noqa: BLE001  keep the last good filter
                pass
            time.sleep(2)

    threading.Thread(target=tombstone_loop, daemon=True).start()
    return create_app(gw, cfg["auth"]["api_keys"])


if os.environ.get("SIGIL_SERVICE") == "gateway":
    app = from_env()
