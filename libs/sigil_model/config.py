"""Model configuration, read from ``configs/model/*.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sigil_core.config import load_yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str = "base"
    id_schema: str = "ids_v1"
    backbone_init: str = "google/t5-v1_1-base"
    d_model: int = 768
    decoder_layers: int = 4
    decoder_heads: int = 12
    decoder_ff: int = 3072
    dropout: float = 0.1
    max_query_tokens: int = 64
    max_doc_tokens: int = 512

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        c = load_yaml(path)
        m = c["model"]
        enc, dec = m["encoder"], m["decoder"]
        return cls(
            name=c["name"], id_schema=c["id_schema"], backbone_init=m["backbone_init"],
            d_model=dec["d_model"], decoder_layers=dec["layers"], decoder_heads=dec["n_heads"],
            decoder_ff=dec["d_ff"], dropout=enc["dropout"], max_query_tokens=enc["max_input_tokens"],
            max_doc_tokens=enc["max_doc_tokens"],
        )
