"""Why did I not get this document? §28.4.

    python scripts/explain_query.py --query "canine post op antibiotics" --doc 6f1e2c44-... \
        --gateway http://localhost:8080 --tenant clinic_4471

Prints the closed-set answer: pruned at level l, reranked below k, filtered by ACL or
tombstone, not in the trie snapshot, cold-start and outranked, or genuinely low relevance.
Needs an operator-scoped key in SIGIL_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    import httpx

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", required=True)
    ap.add_argument("--doc", help="expected doc_uid or semantic id")
    ap.add_argument("--tenant", default=os.environ.get("SIGIL_TENANT", "dev"))
    ap.add_argument("--gateway", default=os.environ.get("SIGIL_GATEWAY", "http://localhost:8080"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="print the full explain payload")
    a = ap.parse_args(argv)
    r = httpx.post(f"{a.gateway}/v1/debug/explain", timeout=30,
                   headers={"Authorization": f"Bearer {os.environ.get('SIGIL_API_KEY', 'dev-operator')}"},
                   json={"query": a.query, "tenant_id": a.tenant, "expected": a.doc, "k": a.k})
    if r.status_code != 200:
        print(f"{r.status_code}: {r.text}")
        return 1
    out = r.json()
    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    for lv in out.get("beam_trace") or []:
        print(f"level {lv['level']}: kept {lv['kept']}, pruned {lv['pruned']}, entropy {lv['entropy']}, top {lv['top'][:3]}")
    print(f"verification: {out['verification']}")
    if chk := out.get("oracle_check"):
        print(f"\n=> {chk['reason']}  " + ", ".join(f"{k}={v}" for k, v in chk.items() if k != "reason"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
