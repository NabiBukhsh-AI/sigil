"""Public API contract. §21.1. Server-side caps are applied after validation, so a client
can ask for anything in range and still gets the server's limits (§27)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrieveOptions(BaseModel):
    beam_width: int = Field(64, ge=1, le=10_000)
    candidate_cap: int = Field(100, ge=1, le=10_000)
    rerank: bool = True
    allow_fallback: bool = True
    diversity_quota: float = Field(0.4, gt=0.0, le=1.0)
    explain: bool = False
    bundle_id: str | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2048)
    k: int = Field(10, ge=1, le=100)
    tenant_id: str
    filters: dict[str, list[str]] = {}
    options: RetrieveOptions = RetrieveOptions()


class ResultItem(BaseModel):
    doc_uid: str
    semantic_id: str
    title: str | None
    snippet: str | None
    scores: dict[str, float | None]
    channel: str
    state: str
    content_uri: str
    content_trust: str
    metadata: dict


class RetrieveResponse(BaseModel):
    trace_id: str
    bundle_id: str
    degraded_mode: bool
    low_confidence: bool
    latency_ms: dict[str, float]
    channels: dict
    results: list[ResultItem]
    trace: dict | None = None


class ExplainRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2048)
    tenant_id: str
    expected: str | None = None  # doc_uid or semantic id: "why did I not get this document?"
    k: int = Field(10, ge=1, le=100)
