"""BM25 over an in-memory inverted index.

One implementation serves four roles: the §25.2 floor baseline, the doc2query-BM25
baseline (same index over document text plus its synthetic queries), the round-trip
consistency filter in query generation (§7.3), BM25 negative mining (§9.1), and the hot
and full lexical channels.

ponytail: numpy CSR postings rebuilt on change. Fine for the hot set (under 3% of the
corpus) and for per-snapshot full-corpus builds; swap in embedded Tantivy (§30) when the
full index outgrows RAM or rebuild time outgrows the snapshot cadence.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


class BM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b
        self.keys: list[str] = []
        self._docs: list[list[str]] = []
        self._pos: dict[str, int] = {}
        self._dead: set[int] = set()
        self._dirty = True

    def __len__(self) -> int:
        return len(self.keys) - len(self._dead)

    def add(self, keys: Iterable[str], texts: Iterable[str]) -> BM25:
        for key, text in zip(keys, texts, strict=True):
            if key in self._pos:  # update = replace
                self._dead.add(self._pos[key])
            self._pos[key] = len(self.keys)
            self.keys.append(key)
            self._docs.append(tokenize(text))
        self._dirty = True
        return self

    def remove(self, keys: Iterable[str]) -> None:
        for key in keys:
            if key in self._pos:
                self._dead.add(self._pos.pop(key))

    def _build(self) -> None:
        vocab: dict[str, int] = {}
        term_ids, doc_ids = [], []
        for d, toks in enumerate(self._docs):
            for t in toks:
                term_ids.append(vocab.setdefault(t, len(vocab)))
            doc_ids.append(np.full(len(toks), d, dtype=np.int64))
        self.vocab = vocab
        n = len(self._docs)
        self.doc_len = np.array([len(t) for t in self._docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if n else 1.0
        if not term_ids:
            self.post_off = np.zeros(len(vocab) + 1, dtype=np.int64)
            self.post_doc = self.post_tf = np.zeros(0)
            self._dirty = False
            return
        pair = np.asarray(term_ids, dtype=np.int64) * max(n, 1) + np.concatenate(doc_ids)
        uniq, tf = np.unique(pair, return_counts=True)
        terms, docs = np.divmod(uniq, max(n, 1))
        self.post_off = np.searchsorted(terms, np.arange(len(vocab) + 1))
        self.post_doc, self.post_tf = docs, tf.astype(np.float64)
        df = np.diff(self.post_off).astype(np.float64)
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5))
        self._dirty = False

    def _scores(self, query: str) -> np.ndarray:
        if self._dirty:
            self._build()
        scores = np.zeros(len(self._docs))
        for t in set(tokenize(query)):
            tid = self.vocab.get(t)
            if tid is None:
                continue
            lo, hi = self.post_off[tid], self.post_off[tid + 1]
            docs, tf = self.post_doc[lo:hi], self.post_tf[lo:hi]
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[docs] / self.avgdl)
            scores += np.bincount(docs, weights=self.idf[tid] * tf * (self.k1 + 1) / (tf + norm),
                                  minlength=len(self._docs))
        if self._dead:
            scores[list(self._dead)] = -np.inf
        return scores

    def search(self, query: str, n: int = 100) -> list[tuple[str, float]]:
        s = self._scores(query)
        live = np.flatnonzero(s > 0)
        top = live[np.lexsort((live, -s[live]))][:n]
        return [(self.keys[i], float(s[i])) for i in top]

    def score(self, query: str, keys: Sequence[str]) -> np.ndarray:
        """BM25 of specific documents, for the ``bm25_norm`` blend feature (§13.3)."""
        s = self._scores(query)
        return np.array([s[self._pos[k]] if k in self._pos else 0.0 for k in keys])


def with_expansions(texts: Mapping[str, str], queries: Mapping[str, list[str]]) -> dict[str, str]:
    """doc2query-BM25 (§25.2): the same synthetic queries SIGIL trains on, indexed as
    document expansions. The most important baseline: if it matches SIGIL, the generative
    model is not earning its cost."""
    return {k: t + "\n" + "\n".join(queries.get(k, [])) for k, t in texts.items()}
