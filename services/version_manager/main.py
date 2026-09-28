"""Version Manager. §17, §20.2 S9, §21.3.

Stores signed bundle manifests and owns the only writable path to the active-bundle
pointer. Activation needs two distinct operators (two-person rule). Rollback is the same
operation pointed at a previous bundle, so it is a pointer flip with no registry change
(§18.5). The last three bundles stay hot-loadable.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException

from sigil_core.bundle import BundleManifest

HOT_BUNDLES = 3


class VersionStore:
    def __init__(self, root: str | Path, signing_key: bytes | None):
        self.root = Path(root)
        (self.root / "bundles").mkdir(parents=True, exist_ok=True)
        self.key = signing_key
        self._pending: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def register(self, m: BundleManifest) -> None:
        if self.key is not None and not m.verify_signature(self.key):
            raise PermissionError("manifest signature invalid")
        p = self.root / "bundles" / f"{m.bundle_id}.json"
        if p.exists():
            raise FileExistsError(f"{m.bundle_id} already registered; bundles are immutable")
        m.save(p)

    def get(self, bundle_id: str) -> BundleManifest:
        p = self.root / "bundles" / f"{bundle_id}.json"
        if not p.exists():
            raise KeyError(bundle_id)
        return BundleManifest.load(p)

    def history(self) -> list[dict]:
        p = self.root / "history.json"
        return json.loads(p.read_text()) if p.exists() else []

    def active(self) -> str | None:
        h = self.history()
        return h[-1]["bundle_id"] if h else None

    def approve(self, bundle_id: str, operator: str) -> dict:
        """First approval records intent; a second, different operator activates."""
        self.get(bundle_id)
        with self._lock:
            approvers = self._pending.setdefault(bundle_id, set())
            approvers.add(operator)
            if len(approvers) < 2:
                return {"bundle_id": bundle_id, "status": "pending", "approvers": sorted(approvers)}
            del self._pending[bundle_id]
            h = self.history()
            h.append({"bundle_id": bundle_id, "approvers": sorted(approvers), "at": datetime.now(UTC).isoformat(),
                      "previous": h[-1]["bundle_id"] if h else None})
            tmp = self.root / ".history.tmp"
            tmp.write_text(json.dumps(h, indent=2))
            os.replace(tmp, self.root / "history.json")
        return {"bundle_id": bundle_id, "status": "active", "approvers": sorted(approvers)}

    def hot(self) -> list[str]:
        seen: list[str] = []
        for e in reversed(self.history()):
            if e["bundle_id"] not in seen:
                seen.append(e["bundle_id"])
        return seen[:HOT_BUNDLES]


def create_app(store: VersionStore, operators: dict[str, str]) -> FastAPI:
    """``operators`` maps API key -> operator name."""
    app = FastAPI(title="sigil-version-manager")

    def operator(key: str) -> str:
        name = operators.get(key.removeprefix("Bearer ").strip())
        if name is None:
            raise HTTPException(403, "operator scope required")
        return name

    @app.post("/v1/models")
    def register(manifest: dict, authorization: str = Header("")):
        operator(authorization)
        try:
            store.register(BundleManifest.from_dict(manifest))
        except PermissionError as e:
            raise HTTPException(403, str(e)) from e
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        return {"registered": manifest["bundle_id"]}

    @app.get("/v1/models:active")
    def active():
        return {"bundle_id": store.active(), "hot": store.hot()}

    @app.get("/v1/models/{bundle_id}")
    def get(bundle_id: str):
        try:
            return store.get(bundle_id).to_dict()
        except KeyError as e:
            raise HTTPException(404, "no such bundle") from e

    @app.post("/v1/models:activate")
    def activate(body: dict, authorization: str = Header("")):
        try:
            return store.approve(body["bundle_id"], operator(authorization))
        except KeyError as e:
            raise HTTPException(404, "no such bundle") from e

    return app


if os.environ.get("SIGIL_SERVICE") == "version_manager":
    key = os.environ.get("SIGIL_BUNDLE_KEY")
    ops = json.loads(os.environ.get("SIGIL_OPERATORS", "{}"))
    app = create_app(VersionStore(os.environ.get("SIGIL_VERSION_ROOT", "artifacts/versions"),
                                  key.encode() if key else None), ops)
