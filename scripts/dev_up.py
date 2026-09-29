"""Run the whole serving plane locally, one process per service, wired over HTTP exactly as
in production (§20.3). Uses the output of scripts/bootstrap_corpus.py.

    python scripts/bootstrap_corpus.py --synthetic 400 --out artifacts/dev
    python scripts/dev_up.py --dev-dir artifacts/dev

    curl -s localhost:8080/v1/retrieve -H 'Authorization: Bearer dev-reader' \
         -d '{"query": "unique7a unique7b topic7word1", "tenant_id": "dev", "k": 5}'

The reranker is left out unless --reranker is given (it downloads a cross-encoder); without
it the gateway serves generative order and says degraded_mode.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PORTS = {"gateway": 8080, "generative": 8081, "reranker": 8082, "hot_lexical": 8083, "full_lexical": 8084,
         "registry": 8085}


def services(dev: Path, with_reranker: bool) -> dict[str, dict]:
    base = {"PYTHONPATH": os.pathsep.join([str(ROOT / "libs"), str(ROOT)]), "SIGIL_CONFIG": str(dev / "serving.yaml")}
    s = {
        "registry": {"SIGIL_SERVICE": "registry", "SIGIL_REGISTRY_SEED": str(dev / "registry.jsonl"),
                     "app": "services.registry.main:app"},
        "hot_lexical": {"SIGIL_SERVICE": "lexical", "SIGIL_LEXICAL_MODE": "hot", "app": "services.lexical.main:app"},
        "full_lexical": {"SIGIL_SERVICE": "lexical", "SIGIL_LEXICAL_MODE": "full",
                         "SIGIL_LEXICAL_SNAPSHOT": str(dev / "lexical_full.jsonl"), "app": "services.lexical.main:app"},
        "generative": {"SIGIL_SERVICE": "generative_retrieval", "SIGIL_BUNDLE": str(dev / "bundle"),
                       "app": "services.generative_retrieval.main:app"},
        "gateway": {"SIGIL_SERVICE": "gateway", "SIGIL_BUNDLE_MANIFEST": str(dev / "bundle" / "manifest.json"),
                    "app": "services.gateway.main:app"},
    }
    if with_reranker:
        s["reranker"] = {"SIGIL_SERVICE": "reranker", "app": "services.reranker.main:app"}
    return {k: {**base, **v} for k, v in s.items()}


def write_config(dev: Path, with_reranker: bool) -> None:
    rel = os.path.relpath(ROOT / "configs" / "serving" / "dev.yaml", dev)
    lines = [f"extends: {Path(rel).as_posix()}", f"decoding_profile: {(ROOT / 'configs/decoding/default.yaml').as_posix()}",
             "endpoints:"]
    for k, p in PORTS.items():
        if k != "gateway":
            lines.append(f"  {k}: http://127.0.0.1:{p}" if k != "reranker" or with_reranker else "  reranker: null")
    (dev / "serving.yaml").write_text("\n".join(lines) + "\n")


def wait(url: str, timeout: float = 90) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-dir", default="artifacts/dev")
    ap.add_argument("--reranker", action="store_true")
    a = ap.parse_args(argv)
    dev = Path(a.dev_dir).resolve()
    write_config(dev, a.reranker)
    procs = []
    order = ["registry", "hot_lexical", "full_lexical", "generative", "reranker", "gateway"]
    specs = services(dev, a.reranker)
    health = {"registry": "/v1/health", "hot_lexical": "/health", "full_lexical": "/health", "generative": "/ready",
              "reranker": "/health", "gateway": "/v1/ready"}
    try:
        for name in [n for n in order if n in specs]:
            env = {**os.environ, **{k: v for k, v in specs[name].items() if k != "app"}}
            procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", specs[name]["app"], "--port",
                                           str(PORTS[name]), "--log-level", "warning"], env=env, cwd=ROOT))
            if not wait(f"http://127.0.0.1:{PORTS[name]}{health[name]}"):
                print(f"{name} did not become ready")
                return 1
            print(f"{name:<13} ready on :{PORTS[name]}")
        print("\nSIGIL is up. Gateway: http://127.0.0.1:8080  (Ctrl-C to stop)")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    sys.exit(main())
