"""S4 Document Registry Service. §12, §20.2.

Internal endpoints serve the gateway's batch resolution, the trie builder's snapshot read,
and the tombstone Bloom push. Public read endpoints are the §21.2 GETs. Writes go through
the ingestion service, which uses the same registry library against the same database, so
identifier assignment stays one serializable transaction either way.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sigil_core.config import load_yaml
from sigil_core.ids import SemanticId
from sigil_registry_client import MemoryRegistry
from sigil_registry_client.http import record_to_wire


def make_registry(cfg: dict):
    r = cfg.get("registry", {})
    schema = cfg.get("id_schema", "ids_v1")
    if r.get("backend") == "postgres":
        from sigil_registry_client.postgres import PostgresRegistry

        return PostgresRegistry(r["dsn"], schema, r.get("redis_url"), r.get("cache_ttl_seconds", 300))
    reg = MemoryRegistry(schema)
    if seed := os.environ.get("SIGIL_REGISTRY_SEED"):  # dev: registry.jsonl from bootstrap_corpus
        import json
        from pathlib import Path

        from sigil_registry_client.http import record_from_wire

        lines = Path(seed).read_text(encoding="utf-8").splitlines()
        reg.restore(record_from_wire(json.loads(x)) for x in lines if x.strip())
    return reg


class ResolveIn(BaseModel):
    semantic_ids: list[str]
    fresh: bool = False


def create_app(registry=None) -> FastAPI:
    registry = registry or make_registry(load_yaml(os.environ.get("SIGIL_CONFIG", "configs/serving/dev.yaml")))
    app = FastAPI(title="sigil-registry")
    app.state.registry = registry

    @app.post("/internal/resolve")
    def resolve(body: ResolveIn):
        if len(body.semantic_ids) > 1000:
            raise HTTPException(429, "batch too large")
        hits = registry.resolve([SemanticId.parse(s) for s in body.semantic_ids], fresh=body.fresh)
        return {"resolved": {str(k): {"record": record_to_wire(v.record), "via_alias": v.via_alias}
                             for k, v in hits.items()}}

    @app.get("/internal/snapshot_ids")
    def snapshot_ids():
        return {"semantic_ids": [str(s) for s in registry.snapshot_ids()], "epoch": registry.epoch}

    @app.get("/internal/tombstones:bloom")
    def bloom():
        return registry.tombstone_bloom().to_wire()

    @app.get("/internal/epoch")
    def epoch():
        return {"epoch": registry.epoch, "hot_set_ratio": registry.hot_set_ratio()}

    @app.get("/v1/documents:byIdentifier")
    def by_identifier(semantic_id: str, schema: str = "ids_v1"):
        sid = SemanticId.parse(semantic_id)
        hit = registry.resolve([sid]).get(sid)
        if hit is None or hit.record.id_schema_version != schema:
            raise HTTPException(404, "no such identifier")
        return {**record_to_wire(hit.record), "via_alias": hit.via_alias}

    @app.get("/v1/documents/{doc_uid}")
    def get(doc_uid: str):
        rec = registry.get(doc_uid)
        if rec is None:
            raise HTTPException(404, "no such document")
        return record_to_wire(rec)

    @app.get("/v1/health")
    def health():
        return {"ok": True}

    return app


if os.environ.get("SIGIL_SERVICE") == "registry":
    app = create_app()
