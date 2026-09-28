"""S3 Reranker Service. §13.2.

[FIXED] A 6-layer, d=384, ~22M parameter MiniLM-class cross-encoder, fp16 or int8, on ONNX
Runtime or TensorRT. Input is ``[CLS] query [SEP] title + rerank_snippet [SEP]`` truncated
to 256 tokens. It is the only component that can rescue the terminal fan-out, so it is not
optional; when it is down the gateway serves generative order with ``degraded_mode=true``.

Scores leave calibrated: isotonic regression fitted offline when a calibration file is
present (§4.5), otherwise a temperature-scaled sigmoid.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sigil_decoding.confidence import Isotonic

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 6 layers, d=384, ~22M params


class CrossEncoder:
    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu", max_len: int = 256):
        from transformers import AutoTokenizer

        self.max_len, self.device = max_len, device
        self.tok = AutoTokenizer.from_pretrained(model if not model.endswith(".onnx") else Path(model).parent)
        if model.endswith(".onnx"):
            import onnxruntime as ort

            self.sess, self.model = ort.InferenceSession(model, providers=ort.get_available_providers()), None
        else:
            import torch
            from transformers import AutoModelForSequenceClassification

            self.sess = None
            self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()
            if device.startswith("cuda"):
                self.model.half()
            self._torch = torch

    def __call__(self, query: str, texts: Sequence[str]) -> np.ndarray:
        enc = self.tok([query] * len(texts), list(texts), truncation="only_second", max_length=self.max_len,
                       padding=True, return_tensors="np" if self.sess else "pt")
        if self.sess:
            feeds = {i.name: enc[i.name] for i in self.sess.get_inputs() if i.name in enc}
            return self.sess.run(None, feeds)[0].reshape(-1)
        with self._torch.inference_mode():
            return self.model(**enc.to(self.device)).logits.float().reshape(-1).cpu().numpy()


class Reranker:
    def __init__(self, score: Callable[[str, Sequence[str]], np.ndarray], max_pairs: int = 400,
                 isotonic: Isotonic | None = None, temperature: float = 1.0):
        self.score, self.max_pairs, self.iso, self.t = score, max_pairs, isotonic, temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        return self.iso(logits) if self.iso else 1 / (1 + np.exp(-logits / self.t))

    def __call__(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if len(texts) > self.max_pairs:  # server-side cap overrides the caller (§20.2 S3)
            raise ValueError(f"{len(texts)} pairs exceeds cap {self.max_pairs}")
        if not texts:
            return np.zeros(0)
        return self.calibrate(np.asarray(self.score(query, texts), dtype=np.float64))


class RerankIn(BaseModel):
    query: str
    texts: list[str]


def create_app(reranker: Reranker) -> FastAPI:
    app = FastAPI(title="sigil-reranker")

    @app.post("/rerank")
    def rerank(body: RerankIn):
        try:
            return {"scores": reranker(body.query, body.texts).tolist()}
        except ValueError as e:
            raise HTTPException(429, str(e)) from e

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


def from_env() -> FastAPI:
    device = "cuda" if os.environ.get("SIGIL_DEVICE", "cpu").startswith("cuda") else "cpu"
    ce = CrossEncoder(os.environ.get("SIGIL_RERANKER_MODEL", DEFAULT_MODEL), device)
    iso = None
    if cal := os.environ.get("SIGIL_RERANKER_CALIBRATION"):
        d = json.loads(Path(cal).read_text())
        iso = Isotonic(tuple(d["xs"]), tuple(d["ys"]))
    return create_app(Reranker(ce, isotonic=iso))


if os.environ.get("SIGIL_SERVICE") == "reranker":
    app = from_env()
