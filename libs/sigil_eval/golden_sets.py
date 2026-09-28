"""Golden set loading and the held-out document slice.

Golden sets are versioned next to the code that gates on them (§19). Formats are TREC:
queries are ``qid<TAB>text``, qrels are ``qid 0 docid grade``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def load_queries(path: str | Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, text = line.split("\t", 1)
            out[qid] = text
    return out


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, _, doc, grade = line.split()
            out.setdefault(qid, {})[doc] = int(grade)
    return out


def in_held_out(doc_uid: str, fraction: float, seed: int) -> bool:
    """[FIXED] §7.5. Stable per document, independent of corpus order and size, so the
    slice does not reshuffle as documents are added. Held-out documents are excluded from
    every query training example but present in the trie."""
    h = hashlib.blake2b(f"{seed}:{doc_uid}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64 < fraction


def held_out_slice(doc_uids: Iterable[str], fraction: float = 0.05, seed: int = 20260901) -> set[str]:
    return {d for d in doc_uids if in_held_out(d, fraction, seed)}
