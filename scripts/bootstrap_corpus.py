"""Bootstrap a deployment from a corpus: fit the quantizer, assign identifiers, build the
first trie, and write a router bundle that serves before any model is trained.

    python scripts/bootstrap_corpus.py --corpus data/corpus.jsonl --out artifacts/dev
    python scripts/bootstrap_corpus.py --synthetic 400 --out artifacts/dev      # demo corpus

corpus.jsonl rows: {"tenant_id", "title", "text", "metadata"?, "acl"?}

Writes to --out:
    bundle/                 manifest.json (backbone "router"), codebooks/, trie/, embedder.json
    registry.jsonl          registry seed for the dev registry service (SIGIL_REGISTRY_SEED)
    lexical_full.jsonl      full lexical snapshot (SIGIL_LEXICAL_SNAPSHOT)
    corpus_export.jsonl     doc_uid, semantic_id, title, text: input to pipelines/dataset_build.py

Phase 0 comes first (§32): measure the hybrid baseline on this corpus before training anything.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from sigil_core.bundle import BundleManifest
from sigil_data.ingestion.chunk import chunk
from sigil_data.ingestion.clean import clean
from sigil_eval.baselines.bm25 import BM25
from sigil_eval.baselines.router import HashEmbedder
from sigil_identifiers.quantizer import RQKMeans, codebook_io, occupancy_report
from sigil_registry_client import MemoryRegistry
from sigil_registry_client.http import record_to_wire
from sigil_registry_client.records import DocState

from services.ingestion.main import ContentStore, Pipeline
from services.trie_builder.main import build_snapshot, corpus_snapshot_name


def bootstrap(docs: list[dict], out: Path, *, embedder: str = "hash", dim: int = 64, k: int = 256,
              bundle_id: str = "bundle_bootstrap", key: bytes | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if embedder == "hash":
        embed, emb_meta = HashEmbedder(dim), {"kind": "hash", "dim": dim}
    else:
        from services.ingestion.main import HFEmbedder

        embed, emb_meta = HFEmbedder(embedder), {"kind": "hf", "model": embedder}
    # ponytail: units are embedded twice (fit, then ingest). Fine for a bootstrap run.
    units = [u.text for d in docs for u in chunk("", clean(d["text"]))]
    k = min(k, max(2, len(units) // 4))  # small corpora cannot populate 256 centroids per level
    q = RQKMeans(k=k, iters=20, restarts=1).fit(embed(units))

    reg = MemoryRegistry()
    store = ContentStore(out / "content")
    pipe = Pipeline(reg, q, embed, BM25(), store)
    for d in docs:
        pipe.add(tenant_id=d.get("tenant_id", "dev"), title=d.get("title"), content=d["text"],
                 metadata=d.get("metadata"), acl=d.get("acl"), initial_load=True)

    b = out / "bundle"
    if b.exists():
        shutil.rmtree(b)
    cb = codebook_io.save(q.codebooks, b / "codebooks", version="cb_v1.0", id_schema=reg.id_schema)
    info = build_snapshot(reg, b / "trie", reg.id_schema)
    (b / "embedder.json").write_text(json.dumps(emb_meta))
    m = BundleManifest(bundle_id=bundle_id, backbone="router", adapter=None, id_schema=reg.id_schema,
                       codebooks=cb.version, codebook_sha256=cb.sha256, trie_snapshot=info["version"],
                       trie_sha256=info["sha256"], corpus_snapshot=corpus_snapshot_name(),
                       corpus_epoch_range=(0, reg.epoch), reranker="rr_v1", dataset="none")
    (m.signed(key) if key else m).save(b / "manifest.json")

    records = reg.records()
    with open(out / "registry.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(record_to_wire(r)) + "\n")
    with open(out / "lexical_full.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            if r.state != DocState.TOMBSTONED:
                f.write(json.dumps({"key": str(r.semantic_id), "text": f"{r.title or ''}\n{r.rerank_snippet or ''}"}) + "\n")
    with open(out / "corpus_export.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            text = store.get(r.content_uri)
            f.write(json.dumps({"doc_uid": r.doc_uid, "semantic_id": str(r.semantic_id), "title": r.title,
                                "text": text, "metadata": r.metadata, "near_dup_group": r.near_dup_group}) + "\n")
    # Serving config for deployment/docker-compose.yaml: in-network names, no reranker by default.
    (out / "serving.compose.yaml").write_text(
        "extends: /app/configs/serving/dev.yaml\nendpoints:\n  generative: http://generative:8080\n"
        "  reranker: null\n  hot_lexical: http://hot_lexical:8080\n  full_lexical: http://full_lexical:8080\n"
        "  registry: http://registry:8080\n")
    return {"bundle": str(b), "documents": len(docs), "units": len(records), "k": k, "trie": info["version"],
            "occupancy": occupancy_report(q.encode(embed(units)), k)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus")
    src.add_argument("--synthetic", type=int, metavar="N", help="generate an N-document demo corpus")
    ap.add_argument("--out", default="artifacts/dev")
    ap.add_argument("--embedder", default="hash", help="'hash' or a HF model id, e.g. BAAI/bge-base-en-v1.5")
    ap.add_argument("--k", type=int, default=256)
    a = ap.parse_args(argv)
    if a.corpus:
        docs = [json.loads(x) for x in Path(a.corpus).read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        from scripts.local_stack import synthetic_corpus

        docs = [{"tenant_id": "dev", "title": f"Doc {i}", "text": t} for _, i, t in synthetic_corpus(16, max(1, a.synthetic // 16))]
    key = os.environ.get("SIGIL_BUNDLE_KEY")
    print(json.dumps(bootstrap(docs, Path(a.out), embedder=a.embedder, k=a.k, key=key.encode() if key else None),
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
