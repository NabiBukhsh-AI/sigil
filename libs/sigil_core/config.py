"""YAML config loading with ``extends:`` and ``${ENV}`` expansion.

Config is part of the bundle and must be reviewable (§30), so there is no other override
mechanism: a derived config names its parent and restates only what differs.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(os.path.expandvars(path.read_text(encoding="utf-8"))) or {}
    parent = data.pop("extends", None)
    return _merge(load_yaml(path.parent / parent), data) if parent else data
