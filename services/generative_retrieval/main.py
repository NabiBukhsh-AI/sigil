"""S2 Generative Retrieval Service. One process holds the model, the beam search, and the
memory-mapped trie (ADR 0003). §20.2.

Readiness verifies the bundle: codebook, trie, and adapter hashes must match the signed
manifest (§17.3), or the pod refuses to become ready. The health endpoint reports the active
snapshot so the gateway never mixes results from pods on different snapshots in one request.
"""

from __future__ import annotations

import os
from dataclasses import replace

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sigil_core.errors import CapExceeded, SigilError
from sigil_decoding import DecodeConfig

from services.generative_retrieval.engine import Engine
from services.generative_retrieval.snapshot_manager import SnapshotManager


class RetrieveIn(BaseModel):
    query: str = Field(min_length=1, max_length=2048)
    beam: int | None = None
    candidate_cap: int | None = None
    widen: bool = False
    explain: bool = False
    probe: str | None = None  # explain only: is this id in the active snapshot?
    bundle_id: str | None = None


class SnapshotIn(BaseModel):
    path: str
    sha256: str


def create_app(engine: Engine, bundle_id: str, decode_cfg: DecodeConfig = DecodeConfig()) -> FastAPI:
    app = FastAPI(title="sigil-generative-retrieval")
    snapshots: SnapshotManager = engine.snapshots

    @app.post("/retrieve")
    def retrieve(body: RetrieveIn):
        if body.bundle_id and body.bundle_id != bundle_id:
            raise HTTPException(409, f"pod serves {bundle_id}, request pinned {body.bundle_id}")
        cfg = replace(decode_cfg, beam=body.beam or decode_cfg.beam,
                      candidate_cap=body.candidate_cap or decode_cfg.candidate_cap)
        try:
            fn = engine.retrieve_widened if body.widen else engine.retrieve
            return {**fn(body.query, cfg, explain=body.explain, probe=body.probe), "bundle_id": bundle_id}
        except CapExceeded as e:
            return JSONResponse({"detail": str(e)}, status_code=429, headers={"Retry-After": "1"})

    @app.post("/snapshot")
    def swap(body: SnapshotIn):
        try:
            return {"active": snapshots.swap(body.path, body.sha256)}
        except SigilError as e:
            raise HTTPException(422, f"refused, previous snapshot kept: {e}") from e

    @app.get("/health")
    def health():
        return {"ok": True, "bundle_id": bundle_id, "trie_snapshot": snapshots.version,
                "trie_sha256": snapshots.sha256, "admission_rejections": engine.rejections}

    @app.get("/ready")
    def ready():
        if not snapshots.ready:
            raise HTTPException(503, "no verified trie snapshot mapped")
        return {"ready": True, "trie_snapshot": snapshots.version}

    return app


def from_bundle() -> FastAPI:
    """Production entrypoint: ``SIGIL_BUNDLE`` points at a bundle directory holding
    ``manifest.json``, ``codebooks/``, ``trie/``, ``model/`` (config.yaml, weights.pt),
    optionally ``adapters/<version>/`` and ``calibration.json``.

    A bundle whose backbone is ``router`` has no trained model yet (a fresh deployment from
    ``scripts/bootstrap_corpus.py``); it serves the frozen quantizer as the router.
    """
    import json
    from pathlib import Path

    from sigil_core.bundle import ArtefactMeta, BundleManifest, check_compatibility
    from sigil_core.config import load_yaml
    from sigil_decoding.confidence import Calibrator
    from sigil_identifiers.quantizer import RQKMeans, codebook_io
    from sigil_trie import TrieSnapshot

    root = Path(os.environ["SIGIL_BUNDLE"])
    m = BundleManifest.load(root / "manifest.json")
    key = os.environ.get("SIGIL_BUNDLE_KEY")
    if key and not m.verify_signature(key.encode()):
        raise SystemExit("bundle signature invalid; refusing to start")
    books, cb = codebook_io.load(root / "codebooks", m.codebook_sha256)
    trie_path = next((root / "trie").glob("*.trie"))
    with TrieSnapshot(trie_path, m.trie_sha256) as t:
        trie_meta = ArtefactMeta(t.version, t.sha256, id_schema=t.id_schema, corpus_snapshot=t.corpus_snapshot)
    adapter_meta = None
    if m.adapter:
        a = json.loads((root / "adapters" / m.adapter / "adapter.json").read_text())
        adapter_meta = ArtefactMeta(a["version"], backbone=a["backbone"], trained_on_snapshot=a["trained_on_snapshot"])
    for w in check_compatibility(m, codebook=cb, trie=trie_meta, adapter=adapter_meta):
        print(w)

    if m.backbone == "router":
        from sigil_eval.baselines.router import HashEmbedder, RouterScorer

        emb = json.loads((root / "embedder.json").read_text())
        if emb["kind"] == "hash":
            embed = HashEmbedder(emb["dim"])
        else:
            from services.ingestion.main import HFEmbedder

            embed = HFEmbedder(emb["model"])
        scorer = RouterScorer(RQKMeans(k=books.shape[1], codebooks=books), embed)
    else:
        import torch

        from sigil_model.config import ModelConfig
        from sigil_model.encoder_decoder import ModelScorer, SigilModel
        from sigil_model.lora import apply_lora, load_adapter
        from sigil_model.tokenization import load_text_tokenizer

        cfg = ModelConfig.from_yaml(root / "model" / "config.yaml")
        model = SigilModel(cfg)
        model.load_state_dict(torch.load(root / "model" / "weights.pt", map_location="cpu"))
        if m.adapter:
            apply_lora(model)
            load_adapter(model, torch.load(root / "adapters" / m.adapter / "adapter.pt", map_location="cpu"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = ModelScorer(model.to(device).half() if device == "cuda" else model,
                             load_text_tokenizer(cfg.backbone_init), device)
    cal_file = root / "calibration.json"
    cal = Calibrator(**json.loads(cal_file.read_text())) if cal_file.exists() else Calibrator()
    snapshots = SnapshotManager(m.id_schema)
    snapshots.swap(trie_path, m.trie_sha256)
    serving = load_yaml(os.environ.get("SIGIL_CONFIG", "configs/serving/dev.yaml"))
    decode_cfg = DecodeConfig.from_yaml_dict(load_yaml(serving["decoding_profile"]))
    return create_app(Engine(scorer, snapshots, cal), m.bundle_id, decode_cfg)


if os.environ.get("SIGIL_SERVICE") == "generative_retrieval":
    app = from_bundle()
