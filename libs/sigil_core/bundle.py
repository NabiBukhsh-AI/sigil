"""Retrieval bundle manifest and the §17.3 compatibility rules.

[FIXED] §17.2: nothing is deployed except a complete, signed bundle. Partial upgrades are
the mechanism by which retrieval systems fail silently.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from sigil_core.errors import BundleIncompatible

SNAPSHOT_GAP_WARN_DAYS = 7
SNAPSHOT_GAP_ALARM_DAYS = 30


@dataclass(frozen=True)
class BundleManifest:
    bundle_id: str
    backbone: str
    adapter: str | None
    id_schema: str
    codebooks: str
    codebook_sha256: str
    trie_snapshot: str
    trie_sha256: str
    corpus_snapshot: str
    corpus_epoch_range: tuple[int, int]
    reranker: str
    dataset: str
    eval_report_uri: str = ""
    created_by: str = "training-pipeline"
    signature: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> BundleManifest:
        d = dict(d)
        d["corpus_epoch_range"] = tuple(d["corpus_epoch_range"])
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> BundleManifest:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["corpus_epoch_range"] = list(self.corpus_epoch_range)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    def _payload(self) -> bytes:
        d = self.to_dict()
        d.pop("signature")
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()

    # ponytail: HMAC shared secret. Switch to ed25519 (cryptography) if signers and
    # verifiers must not share a key.
    def signed(self, key: bytes) -> BundleManifest:
        sig = hmac.new(key, self._payload(), hashlib.sha256).hexdigest()
        return BundleManifest.from_dict({**self.to_dict(), "signature": sig})

    def verify_signature(self, key: bytes) -> bool:
        expected = hmac.new(key, self._payload(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature)


@dataclass(frozen=True)
class ArtefactMeta:
    """What an artefact says about itself, read from its own header or sidecar."""

    version: str
    sha256: str = ""
    id_schema: str | None = None
    backbone: str | None = None  # adapters only
    trained_on_snapshot: str | None = None  # adapters only
    corpus_snapshot: str | None = None  # tries only
    extra: dict = field(default_factory=dict)


def snapshot_date(name: str) -> date:
    """``cs_2026_08_31`` -> date(2026, 8, 31)."""
    y, m, d = name.removeprefix("cs_").split("_")[:3]
    return date(int(y), int(m), int(d))


def check_compatibility(
    m: BundleManifest,
    *,
    codebook: ArtefactMeta,
    trie: ArtefactMeta,
    adapter: ArtefactMeta | None = None,
) -> list[str]:
    """Raise ``BundleIncompatible`` on any hard §17.3 violation; return warnings.

    A model trained on one id schema may never read another schema's trie. Different
    token semantics entirely, so that is a hard failure, never a degradation.
    """
    errors: list[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    expect(codebook.version == m.codebooks, f"codebook {codebook.version} != bundle {m.codebooks}")
    expect(codebook.sha256 == m.codebook_sha256, "codebook sha256 mismatch")
    expect(codebook.id_schema == m.id_schema, f"codebook schema {codebook.id_schema} != {m.id_schema}")
    expect(trie.version == m.trie_snapshot, f"trie {trie.version} != bundle {m.trie_snapshot}")
    expect(trie.sha256 == m.trie_sha256, "trie sha256 mismatch")
    expect(trie.id_schema == m.id_schema, f"trie schema {trie.id_schema} != {m.id_schema}")
    if m.adapter is not None:
        expect(adapter is not None, f"bundle names adapter {m.adapter} but none was loaded")
    if adapter is not None:
        expect(adapter.version == m.adapter, f"adapter {adapter.version} != bundle {m.adapter}")
        expect(adapter.backbone == m.backbone, f"adapter backbone {adapter.backbone} != {m.backbone}")
    if errors:
        raise BundleIncompatible(f"{m.bundle_id}: " + "; ".join(errors))

    warnings: list[str] = []
    if adapter is not None and adapter.trained_on_snapshot and trie.corpus_snapshot:
        gap = (snapshot_date(trie.corpus_snapshot) - snapshot_date(adapter.trained_on_snapshot)).days
        if gap < 0:
            raise BundleIncompatible(f"trie snapshot predates adapter training snapshot by {-gap}d")
        if gap > SNAPSHOT_GAP_ALARM_DAYS:
            warnings.append(f"ALARM: trie is {gap}d ahead of adapter training snapshot")
        elif gap > SNAPSHOT_GAP_WARN_DAYS:
            warnings.append(f"WARN: trie is {gap}d ahead of adapter training snapshot")
    return warnings
