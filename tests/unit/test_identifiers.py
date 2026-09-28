"""Phase 2: determinism, balance, escape, and serializable assignment under concurrency."""

import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from sigil_core.errors import PrefixCapacityExceeded, SnapshotIntegrityError
from sigil_core.ids import ORDINALS_PER_PREFIX
from sigil_identifiers import collision
from sigil_identifiers.assignment import PrefixCounter, escape_rate
from sigil_identifiers.quantizer import codebook_io, occupancy_report
from sigil_identifiers.quantizer.balance import balanced_assign, capacity

from tests.helpers import clustered_embeddings

ROOT = Path(__file__).resolve().parents[2]


def test_balanced_assign_respects_capacity():
    X = clustered_embeddings(4000, clusters=5)
    C = X[:32]
    a = balanced_assign(X, C, max_ratio=3.0)
    assert (a >= 0).all()
    assert np.bincount(a, minlength=32).max() <= capacity(4000, 32, 3.0)


def test_level1_balance_on_clumpy_data(small_quantizer):
    q, X = small_quantizer
    rep = occupancy_report(q.encode(X), q.k)
    assert rep["l1_max_over_mean"] <= 3.0, rep


def test_encode_is_deterministic_across_processes(small_quantizer, tmp_path):
    q, X = small_quantizer
    codebook_io.save(q.codebooks, tmp_path / "cb", version="cb_test", id_schema="ids_v1")
    np.save(tmp_path / "x.npy", X[:200])
    script = (
        "import numpy as np, sys;"
        "from sigil_identifiers.quantizer import RQKMeans, codebook_io;"
        f"cb,_ = codebook_io.load(r'{tmp_path / 'cb'}');"
        "q = RQKMeans(k=16, codebooks=cb);"
        f"np.save(r'{tmp_path / 'out.npy'}', q.encode(np.load(r'{tmp_path / 'x.npy'}')))"
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=ROOT,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "libs")})
    assert np.array_equal(np.load(tmp_path / "out.npy"), q.encode(X[:200]))


def test_codebook_hash_is_enforced(small_quantizer, tmp_path):
    q, _ = small_quantizer
    meta = codebook_io.save(q.codebooks, tmp_path, version="cb_test", id_schema="ids_v1")
    codebook_io.load(tmp_path, meta.sha256)
    with pytest.raises(SnapshotIntegrityError):
        codebook_io.load(tmp_path, "0" * 64)
    np.save(tmp_path / "codebooks.npy", q.codebooks + 1e-3)
    with pytest.raises(SnapshotIntegrityError):
        codebook_io.load(tmp_path)


def test_allocator_escapes_at_256th_and_exhausts():
    c = PrefixCounter()
    ids = [c.allocate((1, 2, 3, 4)) for _ in range(ORDINALS_PER_PREFIX)]
    assert ids[254].tail == (254,) and not ids[254].escaped
    assert ids[255].tail == (255, 0) and ids[255].escaped
    assert len({s.pack() for s in ids}) == ORDINALS_PER_PREFIX
    with pytest.raises(PrefixCapacityExceeded):
        c.allocate((1, 2, 3, 4))
    assert escape_rate(ids) == pytest.approx(510 / 765)


def test_ordinals_never_reused_after_rebuild():
    c = PrefixCounter()
    issued = [c.allocate((0, 0, 0, 0)) for _ in range(3)]
    rebuilt = PrefixCounter.from_issued(issued[:1] + issued[2:])  # middle one deleted
    assert rebuilt.allocate((0, 0, 0, 0)).u == 3


def test_assignment_is_serializable_under_100_writers():
    c = PrefixCounter()
    out, lock = [], threading.Lock()

    def writer():
        got = [c.allocate((9, 9, 9, 9)) for _ in range(7)]
        with lock:
            out.extend(got)

    threads = [threading.Thread(target=writer) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(s.ordinal for s in out) == list(range(700))


def test_reidentification_hysteresis(small_quantizer):
    q, X = small_quantizer
    x = X[0]
    codes = tuple(int(v) for v in q.encode(x[None])[0])
    assert collision.decide(q, x, codes, alias_depth=0).reason == "codes_unchanged"
    far = X[np.argmax(((q.prep(X) - q.prep(x[None])) ** 2).sum(1))]
    d = collision.decide(q, far, codes, alias_depth=0, margin=0.0)
    assert d.reidentify
    assert not collision.decide(q, far, codes, alias_depth=2).reidentify
    assert not collision.decide(q, far, codes, alias_depth=0, margin=1e9).reidentify
