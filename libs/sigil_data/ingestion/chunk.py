"""Chunking to retrieval units. §3.1 I5, §6.4.

[FIXED] The model never sees a unit longer than 512 tokens. Each unit is an independently
identified document with a ``parent_doc_uid`` pointer; the registry, not the model,
reassembles parents.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_SENT = re.compile(r"(?<=[.!?])\s+")


def approx_tokens(text: str) -> int:
    """ponytail: ~1.3 subword tokens per word. Pass the backbone tokenizer's length
    function to ``chunk`` where exactness matters."""
    return int(len(text.split()) * 1.3) + 1


@dataclass(frozen=True)
class Unit:
    parent: str
    ordinal: int
    text: str

    @property
    def key(self) -> str:
        return f"{self.parent}#{self.ordinal}"


def _pieces(text: str, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    out = []
    for para in [p for p in text.split("\n\n") if p.strip()]:
        if count(para) <= max_tokens:
            out.append(para)
            continue
        for sent in _SENT.split(para):
            if count(sent) <= max_tokens:
                out.append(sent)
            else:  # a single monster sentence: hard split on words
                words = sent.split()
                step = max(1, int(max_tokens / 1.3))
                out += [" ".join(words[i : i + step]) for i in range(0, len(words), step)]
    return out


def chunk(parent: str, text: str, max_tokens: int = 512, count: Callable[[str], int] = approx_tokens) -> list[Unit]:
    units, cur = [], ""
    for piece in _pieces(text, max_tokens, count):
        candidate = f"{cur}\n\n{piece}" if cur else piece
        if cur and count(candidate) > max_tokens:
            units.append(cur)
            cur = piece
        else:
            cur = candidate
    if cur:
        units.append(cur)
    return [Unit(parent, i, t) for i, t in enumerate(units)]
