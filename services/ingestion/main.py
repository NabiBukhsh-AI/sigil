"""S5 Corpus Ingestion Service. §3.3, §15.3, §18.3, §21.2.

ADD: parse, clean, quality, dedup by content hash, chunk, embed with the frozen encoder,
quantize with the frozen codebooks, register (ordinal or ESCAPE), index into the hot
lexical channel. The document is live in channel B at once, reaches channel A at the next
trie snapshot through terminal fan-out, and full quality after the next adapter refresh.

UPDATE content re-identifies only past the hysteresis margin; the old id becomes an alias.
UPDATE metadata never touches embeddings or the trie. DELETE tombstones immediately.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sigil_data.ingestion.chunk import chunk
from sigil_data.ingestion.clean import clean, content_hash, html_to_text, quality
from sigil_identifiers import collision
from sigil_identifiers.quantizer import RQKMeans
from sigil_registry_client.records import DocState

Embed = Callable[[list[str]], np.ndarray]
SNIPPET_WORDS = 200  # ~256 tokens, precomputed so reranking never fetches a blob (§22.3)


class ContentStore:
    """Immutable per content hash. ponytail: local directory; point ``root`` at a mounted
    bucket or swap in an S3 client for production."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()
        p = self.root / h[:2] / f"{h}.txt"
        if not p.exists():
            p.parent.mkdir(exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return p.as_uri()

    def purge(self, uri: str) -> None:
        p = Path(uri.removeprefix("file:///").removeprefix("file://"))
        if p.exists():
            p.unlink()


class HFEmbedder:
    """The frozen dense encoder (§3.1 E1-E2). Its identity is part of the id schema: changing
    it is a full retrain (§15.4)."""

    def __init__(self, model: str, device: str = "cpu", max_len: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tok, self.model = AutoTokenizer.from_pretrained(model), AutoModel.from_pretrained(model).to(device).eval()
        self.device, self.max_len, self.torch = device, max_len, torch

    def __call__(self, texts: list[str]) -> np.ndarray:
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            h = self.model(**enc).last_hidden_state
        m = enc["attention_mask"][..., None].float()
        v = (h * m).sum(1) / m.sum(1)
        return self.torch.nn.functional.normalize(v, dim=-1).cpu().numpy()


@dataclass
class Caps:
    hot_alarm: float = 0.05
    hot_block: float = 0.10
    max_tokens: int = 512
    margin: float = 0.05
    max_alias_depth: int = 2


class HotSetOverCap(Exception):
    """§14.2: ingestion blocks when the hot set passes 10% of the corpus. The adapter
    refresh is overdue; letting the hot set grow would make BM25 the system."""


class Pipeline:
    def __init__(self, registry, quantizer: RQKMeans, embed: Embed, hot_index, store: ContentStore,
                 caps: Caps = Caps()):
        self.reg, self.q, self.embed, self.hot, self.store, self.caps = registry, quantizer, embed, hot_index, store, caps
        self.alarms: list[str] = []

    def _check_hot_cap(self) -> None:
        ratio = self.reg.hot_set_ratio()
        if ratio >= self.caps.hot_block and len(self.reg.snapshot_ids()) >= 100:
            raise HotSetOverCap(f"hot set at {ratio:.1%} of corpus; adapter refresh overdue")
        if ratio >= self.caps.hot_alarm:
            self.alarms.append(f"hot_set_ratio={ratio:.3f}")

    def add(self, *, tenant_id: str, content: str, title: str | None = None, metadata: dict | None = None,
            acl: dict | None = None, trusted: bool = True, html: bool = False, initial_load: bool = False) -> dict:
        """``initial_load`` is the bootstrap corpus the first model is trained on: registered
        ACTIVE_LEARNED and kept out of the hot set. Never exposed over HTTP."""
        if not initial_load:
            self._check_hot_cap()
        text = clean(html_to_text(content) if html else content)
        ok, why = quality(text)
        if not ok:
            raise ValueError(f"rejected by quality filter: {why}")
        units = chunk("", text, self.caps.max_tokens)
        vectors = self.embed([u.text for u in units])
        codes = self.q.encode(vectors)
        parent = str(uuid.uuid4())
        state = DocState.QUARANTINED if not trusted else (  # §27 poisoning control
            DocState.ACTIVE_LEARNED if initial_load else DocState.ACTIVE_COLD_START)
        out = []
        for i, (u, c) in enumerate(zip(units, codes, strict=True)):
            rec = self.reg.create(
                tenant_id=tenant_id, codes=tuple(int(x) for x in c), content_hash=content_hash(f"{title}\n{u.text}"),
                content_uri=self.store.put(u.text), title=title, rerank_snippet=" ".join(u.text.split()[:SNIPPET_WORDS]),
                parent_doc_uid=parent, acl=acl, metadata={**(metadata or {}), "chunk": i},
                content_trust="trusted" if trusted else "untrusted", state=state,
                doc_uid=str(uuid.uuid5(uuid.UUID(parent), str(i))),
            )
            if rec.state == DocState.ACTIVE_COLD_START:
                self.hot.add([str(rec.semantic_id)], [f"{title or ''}\n{u.text}"])
            out.append({"doc_uid": rec.doc_uid, "semantic_id": str(rec.semantic_id), "state": rec.state.value})
        return {"doc_uid": parent, "chunks": out,
                "retrievable": {"hot_lexical": trusted, "generative": False, "eta_generative_seconds": 900},
                "corpus_epoch": self.reg.epoch}

    def update(self, parent: str, *, content: str, title: str | None = None) -> dict:
        """Full replace. Chunks are matched by position; surplus old chunks are tombstoned
        and extra new ones created."""
        old = [r for r in self.reg.children(parent) if r.state != DocState.TOMBSTONED]
        if not old:
            raise KeyError(parent)
        units = chunk("", clean(content), self.caps.max_tokens)
        vectors = self.embed([u.text for u in units])
        out = []
        for i, (u, v) in enumerate(zip(units, vectors, strict=True)):
            title_i = title if title is not None else old[0].title
            h = content_hash(f"{title_i}\n{u.text}")
            if i < len(old):
                rec = old[i]
                depth = rec.metadata.get("alias_depth", 0)
                d = collision.decide(self.q, v, rec.semantic_id.codes, alias_depth=depth, margin=self.caps.margin,
                                     max_alias_depth=self.caps.max_alias_depth)
                new = self.reg.update_content(rec.doc_uid, content_hash=h, content_uri=self.store.put(u.text),
                                              rerank_snippet=" ".join(u.text.split()[:SNIPPET_WORDS]),
                                              new_codes=d.new_codes if d.reidentify else None)
                if d.reidentify:
                    self.reg.patch(rec.doc_uid, metadata={"alias_depth": depth + 1})
                self.hot.remove([str(rec.semantic_id)])
                if new.state == DocState.ACTIVE_COLD_START:
                    self.hot.add([str(new.semantic_id)], [f"{title_i or ''}\n{u.text}"])
                out.append({"doc_uid": new.doc_uid, "semantic_id": str(new.semantic_id), "reidentified": d.reidentify,
                            "reason": d.reason})
            else:
                rec = self.reg.create(
                    tenant_id=old[0].tenant_id, codes=tuple(int(x) for x in self.q.encode(v[None])[0]),
                    content_hash=h, content_uri=self.store.put(u.text), title=title_i,
                    rerank_snippet=" ".join(u.text.split()[:SNIPPET_WORDS]), parent_doc_uid=parent, acl=old[0].acl,
                    metadata={**old[0].metadata, "chunk": i}, doc_uid=str(uuid.uuid5(uuid.UUID(parent), str(i))),
                )
                self.hot.add([str(rec.semantic_id)], [f"{title_i or ''}\n{u.text}"])
                out.append({"doc_uid": rec.doc_uid, "semantic_id": str(rec.semantic_id), "reidentified": False,
                            "reason": "new_chunk"})
        for rec in old[len(units):]:
            self.delete_unit(rec.doc_uid)
        return {"doc_uid": parent, "chunks": out, "corpus_epoch": self.reg.epoch}

    def delete_unit(self, doc_uid: str, hard: bool = False, actor: str = "api") -> None:
        rec = self.reg.get(doc_uid)
        self.hot.remove([str(rec.semantic_id)])
        if hard:
            self.store.purge(rec.content_uri)
            self.reg.hard_delete(doc_uid, actor)
        else:
            self.reg.tombstone(doc_uid, actor)

    def delete(self, parent: str, hard: bool = False, actor: str = "api") -> int:
        units = [r for r in self.reg.children(parent) if r.state != DocState.TOMBSTONED or hard]
        for r in units:
            self.delete_unit(r.doc_uid, hard, actor)
        return len(units)


# -- HTTP (§21.2) ---------------------------------------------------------------------------


class CreateIn(BaseModel):
    tenant_id: str
    title: str | None = None
    content: str = Field(min_length=1, max_length=5_000_000)
    metadata: dict = {}
    acl: dict = {}
    trusted: bool = True
    html: bool = False


class ReplaceIn(BaseModel):
    content: str = Field(min_length=1, max_length=5_000_000)
    title: str | None = None


class PatchIn(BaseModel):
    title: str | None = None
    metadata: dict | None = None
    acl: dict | None = None


def create_app(p: Pipeline, write_keys: dict[str, dict]) -> FastAPI:
    from fastapi import Depends, Header

    app = FastAPI(title="sigil-ingestion")

    def caller(authorization: str = Header("")) -> dict:
        c = write_keys.get(authorization.removeprefix("Bearer ").strip())
        if c is None or "write" not in c.get("scopes", []):
            raise HTTPException(403, "write scope required")
        return c

    def owned(parent: str, c: dict):
        kids = p.reg.children(parent)
        if not kids:
            raise HTTPException(404, "no such document")
        if kids[0].tenant_id != c["tenant"]:
            raise HTTPException(403, "tenant mismatch")
        return kids

    @app.post("/v1/documents", status_code=201)
    def create(body: CreateIn, c: dict = Depends(caller)):
        if body.tenant_id != c["tenant"]:
            raise HTTPException(403, "tenant mismatch")
        try:
            return p.add(tenant_id=body.tenant_id, content=body.content, title=body.title, metadata=body.metadata,
                         acl=body.acl, trusted=body.trusted and c.get("trusted_source", True), html=body.html)
        except HotSetOverCap as e:
            raise HTTPException(503, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.put("/v1/documents/{doc_uid}")
    def replace(doc_uid: str, body: ReplaceIn, c: dict = Depends(caller)):
        owned(doc_uid, c)
        return p.update(doc_uid, content=body.content, title=body.title)

    @app.patch("/v1/documents/{doc_uid}")
    def patch(doc_uid: str, body: PatchIn, c: dict = Depends(caller)):
        kids = owned(doc_uid, c)
        for r in kids:
            p.reg.patch(r.doc_uid, title=body.title, metadata=body.metadata, acl=body.acl, actor=c["tenant"])
        return {"doc_uid": doc_uid, "updated": len(kids)}

    @app.delete("/v1/documents/{doc_uid}")
    def delete(doc_uid: str, hard: bool = False, c: dict = Depends(caller)):
        owned(doc_uid, c)
        if hard and "erase" not in c.get("scopes", []):
            raise HTTPException(403, "hard delete requires elevated scope")
        return {"doc_uid": doc_uid, "deleted": p.delete(doc_uid, hard, actor=c["tenant"])}

    @app.get("/v1/health")
    def health():
        return {"ok": True, "hot_set_ratio": p.reg.hot_set_ratio(), "alarms": p.alarms[-10:]}

    return app


def from_env() -> FastAPI:
    from sigil_core.config import load_yaml
    from sigil_identifiers.quantizer import codebook_io

    from services.registry.main import make_registry

    cfg = load_yaml(os.environ.get("SIGIL_CONFIG", "configs/serving/dev.yaml"))
    ids = load_yaml(os.environ.get("SIGIL_ID_SCHEMA", "configs/identifiers/ids_v1.yaml"))
    books, _ = codebook_io.load(os.environ["SIGIL_CODEBOOKS"], os.environ.get("SIGIL_CODEBOOK_SHA256"))
    q = RQKMeans.from_config(ids)
    q.codebooks = books

    import httpx

    hot_url = cfg["endpoints"]["hot_lexical"]

    class HotClient:
        def add(self, keys, texts):
            httpx.post(f"{hot_url}/index", json={"docs": [{"key": k, "text": t} for k, t in zip(keys, texts)]})

        def remove(self, keys):
            httpx.post(f"{hot_url}/remove", json={"keys": list(keys)})

    lc = cfg["lifecycle"]
    pipeline = Pipeline(make_registry(cfg), q, HFEmbedder(ids["quantizer"]["embedding_model"]), HotClient(),
                        ContentStore(os.environ.get("SIGIL_CONTENT_ROOT", "artifacts/content")),
                        Caps(hot_alarm=lc["hot_set_alarm_ratio"], hot_block=lc["hot_set_block_ingest_ratio"]))
    return create_app(pipeline, cfg["auth"]["api_keys"])


if os.environ.get("SIGIL_SERVICE") == "ingestion":
    app = from_env()
