"""S6 Trie Builder. §18.3, §20.2.

Every 15 minutes or 50k deltas, whichever comes first: read the registry snapshot, compile,
validate against the same id set, publish content-addressed, and tell every generative pod
to swap. A snapshot that fails validation is deleted, never published.

    python -m services.trie_builder.main --out artifacts/tries --once
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from sigil_core.errors import SnapshotIntegrityError
from sigil_trie import TrieSnapshot, write
from sigil_trie.validate import validate


def corpus_snapshot_name(t: datetime | None = None) -> str:
    return (t or datetime.now(UTC)).strftime("cs_%Y_%m_%d")


def build_snapshot(registry, out_dir: str | Path, id_schema: str) -> dict:
    t0 = time.perf_counter()
    epoch = registry.epoch
    ids = registry.snapshot_ids()
    path, sha = write(ids, out_dir, id_schema=id_schema, corpus_snapshot=corpus_snapshot_name())
    with TrieSnapshot(path, sha) as trie:
        problems = validate(trie, ids)
        ids_sha = trie.header["ids_sha256"]
    if problems:
        Path(path).unlink(missing_ok=True)
        raise SnapshotIntegrityError(f"snapshot failed validation, not published: {problems}")
    info = {"path": str(path), "sha256": sha, "version": f"trie_{sha[:12]}", "n_ids": len(ids), "epoch": epoch,
            "ids_sha256": ids_sha, "id_schema": id_schema, "built_at": datetime.now(UTC).isoformat(),
            "build_seconds": round(time.perf_counter() - t0, 3)}
    tmp = Path(out_dir) / ".latest.json.tmp"
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, Path(out_dir) / "latest.json")  # atomic pointer
    return info


def publish(info: dict, pod_urls: list[str]) -> dict[str, str]:
    """Ask every generative pod to map and verify the new snapshot. Pods that refuse keep
    their previous snapshot; the gateway never mixes snapshots within a request."""
    import httpx

    acks = {}
    for url in pod_urls:
        try:
            r = httpx.post(f"{url}/snapshot", json={"path": info["path"], "sha256": info["sha256"]}, timeout=30)
            acks[url] = r.json().get("active", f"refused: {r.text[:200]}")
        except Exception as e:  # noqa: BLE001
            acks[url] = f"unreachable: {e!r}"[:200]
    return acks


def run(registry, out_dir: str, id_schema: str, pod_urls: list[str], every_s: int = 900, max_delta: int = 50_000,
        poll_s: int = 10, once: bool = False) -> None:
    last_epoch, last_build = -1, 0.0
    while True:
        epoch = registry.epoch
        due = time.time() - last_build >= every_s or epoch - last_epoch >= max_delta
        if epoch != last_epoch and (due or once):
            info = build_snapshot(registry, out_dir, id_schema)
            info["acks"] = publish(info, pod_urls)
            print(json.dumps(info))
            last_epoch, last_build = epoch, time.time()
        if once:
            return
        time.sleep(poll_s)


def main(argv=None) -> None:
    from sigil_core.config import load_yaml
    from sigil_registry_client.http import HttpRegistry

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("SIGIL_CONFIG", "configs/serving/dev.yaml"))
    ap.add_argument("--out", default="artifacts/tries")
    ap.add_argument("--pods", nargs="*", default=[])
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    run(HttpRegistry(cfg["endpoints"]["registry"]), a.out, cfg["id_schema"], a.pods or [cfg["endpoints"]["generative"]],
        every_s=cfg["trie"]["snapshot_minutes"] * 60, once=a.once)


if __name__ == "__main__":
    main()
