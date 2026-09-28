"""Check a bundle directory against its manifest before it goes anywhere. §17.2, §17.3.

    python scripts/verify_bundle.py --bundle artifacts/bundles/bundle_2026_09_02_a

Verifies the signature (when SIGIL_BUNDLE_KEY is set), the codebook and trie hashes, schema
agreement, and adapter/backbone compatibility. Exit 0 means a pod would become ready on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sigil_core.bundle import ArtefactMeta, BundleManifest, check_compatibility
from sigil_core.errors import SigilError
from sigil_identifiers.quantizer import codebook_io
from sigil_trie import TrieSnapshot


def verify(bundle: str | Path, key: bytes | None = None) -> list[str]:
    root = Path(bundle)
    m = BundleManifest.load(root / "manifest.json")
    if key is not None and not m.verify_signature(key):
        raise SigilError("manifest signature invalid")
    _, cb = codebook_io.load(root / "codebooks", m.codebook_sha256)
    trie_path = next((root / "trie").glob("*.trie"))
    with TrieSnapshot(trie_path, m.trie_sha256) as t:
        trie = ArtefactMeta(t.version, t.sha256, id_schema=t.id_schema, corpus_snapshot=t.corpus_snapshot)
    adapter = None
    if m.adapter:
        a = json.loads((root / "adapters" / m.adapter / "adapter.json").read_text())
        adapter = ArtefactMeta(a["version"], backbone=a["backbone"], trained_on_snapshot=a["trained_on_snapshot"])
    return check_compatibility(m, codebook=cb, trie=trie, adapter=adapter)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    a = ap.parse_args(argv)
    key = os.environ.get("SIGIL_BUNDLE_KEY")
    try:
        warnings = verify(a.bundle, key.encode() if key else None)
    except (SigilError, FileNotFoundError, StopIteration) as e:
        print(f"REFUSE: {e}")
        return 1
    for w in warnings:
        print(w)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
