"""S7 Hot Lexical and S8 Full Lexical. Same binary, different scope and caps. §14.2, §16.

hot:  BM25 over ACTIVE_COLD_START documents only. Near-real-time adds; a new document is
      searchable here in well under the 60 s goal. Hot set must stay under 3% of corpus.
full: BM25 over the whole corpus, loaded from a snapshot. Invoked on trigger only, and
      must not exceed 15% of queries (fallback creep, ADR 0005).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sigil_eval.baselines.bm25 import BM25


class SearchIn(BaseModel):
    query: str
    n: int = 25


class Doc(BaseModel):
    key: str
    text: str


class IndexIn(BaseModel):
    docs: list[Doc]


class RemoveIn(BaseModel):
    keys: list[str]


def create_app(mode: str, index: BM25 | None = None) -> FastAPI:
    if mode not in ("hot", "full"):
        raise ValueError(mode)
    index = index or BM25()
    lock = threading.Lock()
    app = FastAPI(title=f"sigil-lexical-{mode}")

    @app.post("/search")
    def search(body: SearchIn):
        with lock:
            hits = index.search(body.query, min(body.n, 400))
        return {"hits": [{"key": k, "score": s} for k, s in hits]}

    @app.post("/index")
    def add(body: IndexIn):
        if mode != "hot":
            raise HTTPException(409, "full index is snapshot-built; rebuild instead of adding")
        with lock:
            index.add([d.key for d in body.docs], [d.text for d in body.docs])
            index.search("", 1)  # rebuild now, not on the next query
        return {"size": len(index)}

    @app.post("/remove")
    def remove(body: RemoveIn):
        with lock:
            index.remove(body.keys)
        return {"size": len(index)}

    @app.get("/health")
    def health():
        return {"ok": True, "mode": mode, "size": len(index)}

    return app


def from_env() -> FastAPI:
    mode = os.environ.get("SIGIL_LEXICAL_MODE", "hot")
    index = BM25()
    if snap := os.environ.get("SIGIL_LEXICAL_SNAPSHOT"):  # JSONL of {"key", "text"}
        rows = [json.loads(line) for line in Path(snap).read_text(encoding="utf-8").splitlines() if line.strip()]
        index.add([r["key"] for r in rows], [r["text"] for r in rows])
    return create_app(mode, index)


if os.environ.get("SIGIL_SERVICE") == "lexical":
    app = from_env()
