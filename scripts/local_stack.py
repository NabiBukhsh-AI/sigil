"""A complete SIGIL deployment in one process: registry, ingestion, trie builder,
generative service, hot and full lexical, reranker stand-in, gateway.

Used by the churn simulator (Experiment 8), lifecycle and quality tests, and benchmarks.
The generative channel is the untrained quantizer router unless a scorer is passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sigil_decoding import DecodeConfig, Scorer
from sigil_eval.baselines.bm25 import BM25
from sigil_eval.baselines.router import HashEmbedder, RouterScorer, overlap_reranker
from sigil_identifiers.quantizer import RQKMeans
from sigil_registry_client import MemoryRegistry
from sigil_registry_client.records import DocState, Principal

from services.gateway.main import Gateway, GatewayConfig, local_backends
from services.gateway.schemas import RetrieveRequest
from services.generative_retrieval.engine import Engine
from services.generative_retrieval.snapshot_manager import SnapshotManager
from services.ingestion.main import ContentStore, Pipeline
from services.trie_builder.main import build_snapshot


def synthetic_corpus(topics: int = 16, per_topic: int = 10, seed: int = 0) -> list[tuple[int, int, str]]:
    """(topic, doc index, text). Each doc: 30 topic words plus 3 words unique to it."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(topics * per_topic):
        t = i % topics
        vocab = [f"topic{t}word{j}" for j in range(12)]
        out.append((t, i, " ".join([*rng.choice(vocab, 30), f"unique{i}a", f"unique{i}b", f"unique{i}c"])))
    return out


def query_for(t: int, i: int) -> str:
    return f"unique{i}a unique{i}b topic{t}word1 topic{t}word2"


@dataclass
class Stack:
    workdir: Path
    reg: MemoryRegistry
    pipe: Pipeline
    snaps: SnapshotManager
    engine: Engine
    gw: Gateway
    parent: dict[int, str] = field(default_factory=dict)
    last_snapshot: dict = field(default_factory=dict)

    @property
    def backends(self):
        return self.gw.b

    def rebuild(self) -> dict:
        """Next trie snapshot, plus the full-lexical index rebuilt from the same registry state."""
        info = build_snapshot(self.reg, self.workdir / "tries", self.reg.id_schema)
        self.snaps.swap(info["path"], info["sha256"])
        full = BM25()
        live = {str(r.semantic_id): f"{r.title}\n{r.rerank_snippet}" for r in self.reg.records()
                if r.state not in (DocState.TOMBSTONED, DocState.QUARANTINED)}
        full.add(live.keys(), live.values())
        self.gw.b.full_lexical = full.search
        self.gw.refresh_tombstones()
        self.last_snapshot = info
        return info

    def close(self) -> None:
        self.snaps.close()
        self.gw.pool.shutdown(wait=False)

    def ask(self, query: str, tenant: str = "clinic", k: int = 10, principal: Principal | None = None, **opts) -> dict:
        req = RetrieveRequest(query=query, tenant_id=tenant, k=k, options=opts or {})
        return self.gw.retrieve(req, principal or Principal(tenant))[0]


def build(docs, workdir: str | Path, *, k: int = 16, dim: int = 64, tenant: str = "clinic",
          decode: DecodeConfig = DecodeConfig(beam=16, prefixes_expanded=16), scorer: Scorer | None = None) -> Stack:
    workdir = Path(workdir)
    embed = HashEmbedder(dim)
    q = RQKMeans(k=k, iters=10, restarts=1).fit(embed([d for _, _, d in docs]))
    reg, hot = MemoryRegistry(), BM25()
    pipe = Pipeline(reg, q, embed, hot, ContentStore(workdir / "content"))
    parent = {i: pipe.add(tenant_id=tenant, title=f"Doc {i}", content=text, initial_load=True)["doc_uid"]
              for _, i, text in docs}
    snaps = SnapshotManager(reg.id_schema)
    engine = Engine(scorer or RouterScorer(q, embed), snaps)
    cfg = GatewayConfig(bundle_id="bundle_local", id_schema=reg.id_schema, decode=decode,
                        deadlines_ms={"generative": 10_000, "lexical": 10_000, "rerank": 10_000})
    gw = Gateway(cfg, local_backends(engine, hot, BM25(), reg, overlap_reranker))
    stack = Stack(workdir, reg, pipe, snaps, engine, gw, parent)
    stack.rebuild()
    return stack
