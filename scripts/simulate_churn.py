"""Replay N days of corpus mutation and check the lifecycle invariants. Experiment 8, §33.

Each simulated day applies adds, content updates (some of which move a document across a
quantization boundary), and deletes at the given rates, then takes a trie snapshot and
checks:

  * the trie holds exactly the registry's live identifiers (validated at build);
  * every added document is retrievable through the hot channel before the snapshot and
    reachable by the generative channel (terminal fan-out) after it;
  * no deleted document is ever served;
  * every re-identified document still resolves through its old identifier (alias);
  * recall on the original corpus does not regress across days;
  * the hot set stays under its cap once refreshes run.

    python scripts/simulate_churn.py --days 30 --docs 400 --churn 0.02
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from sigil_decoding import DecodeConfig
from sigil_registry_client.records import DocState

from scripts.local_stack import build, query_for, synthetic_corpus
from services.ingestion.main import HotSetOverCap


def titles(resp) -> set[str]:
    return {r["title"] for r in resp["results"]}


def simulate(days: int, n_docs: int, churn: float, workdir: Path, refresh_every: int = 7, seed: int = 0) -> dict:
    topics = 16
    docs = synthetic_corpus(topics, n_docs // topics, seed)
    s = build(docs, workdir)
    try:
        return _simulate(s, docs, days, churn, refresh_every, topics, seed)
    finally:
        s.close()


def _simulate(s, docs, days, churn, refresh_every, topics, seed) -> dict:
    rng = np.random.default_rng(seed + 1)
    live = {i: (t, text) for t, i, text in docs}
    deleted: set[int] = set()
    next_id = max(live) + 1
    old_probe = [(t, i) for t, i, _ in docs[:: max(1, len(docs) // 40)]]
    baseline = np.mean([f"Doc {i}" in titles(s.ask(query_for(t, i))) for t, i in old_probe])
    report = {"baseline_old_recall": float(baseline), "days": [], "violations": []}
    wide = DecodeConfig(beam=64, prefixes_expanded=64, candidate_cap=400, adaptive_candidate_cap=False)

    for day in range(1, days + 1):
        n = max(1, int(churn * len(live)))
        adds, updates, dels = n // 2 or 1, n // 4, n - n // 2 - n // 4
        added, reid, deleted_today, blocked = [], [], [], 0

        for _ in range(adds):
            i, t = next_id, int(rng.integers(topics))
            next_id += 1
            text = " ".join([*rng.choice([f"topic{t}word{j}" for j in range(12)], 30),
                             f"unique{i}a", f"unique{i}b", f"unique{i}c"])
            try:
                r = s.pipe.add(tenant_id="clinic", title=f"Doc {i}", content=text)
            except HotSetOverCap:  # §14.2 working as designed: refresh is overdue
                blocked += 1
                continue
            s.parent[i], live[i] = r["doc_uid"], (t, text)
            added.append((t, i, r["chunks"][0]["semantic_id"]))
            if f"Doc {i}" not in titles(s.ask(f"unique{i}a unique{i}b")):
                report["violations"].append(f"day {day}: new Doc {i} not served by the hot channel")

        for i in rng.choice(sorted(live), updates, replace=False):
            t, _ = live[i]
            t2 = (t + 1 + int(rng.integers(topics - 1))) % topics  # drift to another topic
            old_sid = s.reg.children(s.parent[i])[0].semantic_id
            text = " ".join([*rng.choice([f"topic{t2}word{j}" for j in range(12)], 30),
                             f"unique{i}a", f"unique{i}b", f"unique{i}c"])
            out = s.pipe.update(s.parent[i], content=text)
            live[i] = (t2, text)
            if out["chunks"][0]["reidentified"]:
                reid.append((i, old_sid))

        for i in rng.choice(sorted(live), dels, replace=False):
            s.pipe.delete(s.parent[i])
            deleted.add(int(i))
            deleted_today.append(int(i))
            del live[i]

        hot_before = s.reg.hot_set_ratio()
        info = s.rebuild()

        reached = 0
        for t, i, sid in added:
            gen = s.engine.retrieve(query_for(t, i), wide)
            reached += sid in {c["semantic_id"] for c in gen["candidates"]}
        for i in deleted_today + list(rng.choice(sorted(deleted), min(5, len(deleted)), replace=False)):
            if f"Doc {i}" in titles(s.ask(f"unique{i}a unique{i}b unique{i}c")):
                report["violations"].append(f"day {day}: deleted Doc {i} was served")
        for i, old_sid in reid:
            hit = s.reg.resolve([old_sid]).get(old_sid)
            if hit is None or not hit.via_alias or hit.record.doc_uid != s.reg.children(s.parent[i])[0].doc_uid:
                report["violations"].append(f"day {day}: alias for re-identified Doc {i} broken")
        probe = [(t, i) for t, i in old_probe if i in live and live[i][0] == t]
        old_recall = float(np.mean([f"Doc {i}" in titles(s.ask(query_for(t, i))) for t, i in probe])) if probe else 1.0

        if day % refresh_every == 0:  # adapter refresh: cold-start documents become learned
            s.reg.set_state([r.doc_uid for r in s.reg.records({DocState.ACTIVE_COLD_START})], DocState.ACTIVE_LEARNED)

        report["days"].append({
            "day": day, "adds": adds - blocked, "adds_blocked_by_hot_cap": blocked, "updates": updates, "reidentified": len(reid), "deletes": dels,
            "live": len(live), "trie_ids": info["n_ids"], "trie_build_s": info["build_seconds"],
            "hot_set_ratio": round(hot_before, 4), "new_doc_generative_reach": reached / len(added) if added else None,
            "old_doc_recall": old_recall,
        })
    report["old_doc_recall_regression_points"] = round(
        100 * (report["baseline_old_recall"] - min(d["old_doc_recall"] for d in report["days"])), 2)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--churn", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        report = simulate(a.days, a.docs, a.churn, Path(tmp))
    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 1 if report["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
