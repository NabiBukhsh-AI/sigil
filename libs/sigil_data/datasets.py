"""Versioned training datasets. §7.1, §19.

A dataset version is a directory of Parquet shards plus a manifest recording the content
hash, the corpus snapshot and id schema it was built against, and provenance counts per
source (§27: every dataset version records source provenance counts, so a poisoning
investigation can say what went in).

ponytail: plain Parquet + manifest. Move to Iceberg (§30) when time travel or schema
evolution over 100M-row tables is actually needed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np

REQUIRED = ("input_text", "target_codes", "doc_id", "source", "weight", "corpus_snapshot", "id_schema")
SHARD_ROWS = 1_000_000


def _check(row: dict) -> dict:
    missing = [k for k in REQUIRED if k not in row]
    if missing:
        raise ValueError(f"dataset row missing {missing}: {row.get('doc_id')}")
    if len(row["target_codes"]) != 5:
        raise ValueError(f"target_codes must be [c1,c2,c3,c4,u]: {row['doc_id']}")
    return {"hard_negatives": [], "hard_negative_codes": [], **row}


def write(rows: Iterable[dict], directory: str | Path, *, version: str) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = Path(directory) / version
    d.mkdir(parents=True, exist_ok=False)  # versions are immutable
    h = hashlib.sha256()
    prov: Counter = Counter()
    snaps, schemas, n, shard, buf = set(), set(), 0, 0, []

    def flush():
        nonlocal shard, buf
        if buf:
            pq.write_table(pa.Table.from_pylist(buf), d / f"part-{shard:05d}.parquet")
            shard, buf = shard + 1, []

    for row in rows:
        row = _check(row)
        h.update(json.dumps(row, sort_keys=True).encode())
        prov[row["source"]] += 1
        snaps.add(row["corpus_snapshot"])
        schemas.add(row["id_schema"])
        buf.append(row)
        n += 1
        if len(buf) >= SHARD_ROWS:
            flush()
    flush()
    if len(schemas) > 1:
        raise ValueError(f"mixed id schemas in one dataset: {schemas}")
    manifest = {"version": version, "rows": n, "sha256": h.hexdigest(), "provenance": dict(prov),
                "corpus_snapshots": sorted(snaps), "id_schema": next(iter(schemas), None), "shards": shard}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read(directory: str | Path) -> Iterator[dict]:
    import pyarrow.parquet as pq

    for p in sorted(Path(directory).glob("part-*.parquet")):
        yield from pq.read_table(p).to_pylist()


def batches(rows: Sequence[dict], batch_size: int, seed: int = 0) -> Iterator[list[dict]]:
    """Endless shuffled epochs; the training loop stops on step count."""
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(rows))
        for s in range(0, len(order) - batch_size + 1, batch_size):
            yield [rows[i] for i in order[s : s + batch_size]]
