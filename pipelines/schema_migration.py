"""Identifier-schema migration (codebook refit). §15.6, §35B R4, ADR 0004.

The single most dangerous operation in the system. Follow docs/runbooks/schema-migration.md.

    # 1. plan: refit (or reuse frozen codebooks with --codebooks) and assign every document
    python pipelines/schema_migration.py plan --corpus export/corpus.jsonl --embeddings export/embeddings.npy \
        --ids-config configs/identifiers/ids_v2.yaml --codebook-version cb_v2.0 --out migrations/ids_v2/plan

    # 2. stage the dual identifier set: new-schema ids resolve through aliases
    python pipelines/schema_migration.py apply --stage prepopulate --plan migrations/ids_v2/plan --dsn $DSN

    # 5. cutover (two-person rule)
    python pipelines/schema_migration.py apply --stage cutover --plan migrations/ids_v2/final --dsn $DSN \
        --approved-by alice --approved-by bob

The plan directory holds: codebooks/ (hashed), trie/ (new schema), mapping.csv (doc_uid, new id
as bytea hex), counters.csv (per-prefix next ordinal), plan.json (gates and statistics), and the
exact SQL each stage runs (prepopulate.sql, cutover.sql) for review. Every stage is one
transaction and set-based: the mapping is bulk-loaded with COPY, never one statement per row.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
from sigil_core.config import load_yaml
from sigil_core.ids import SemanticId
from sigil_identifiers.assignment import PrefixCounter, escape_rate
from sigil_identifiers.quantizer import RQKMeans, codebook_io, occupancy_report
from sigil_trie import write

from services.trie_builder.main import corpus_snapshot_name

LOAD_MAP = "-- load: mapping.csv -> migration_map"
LOAD_COUNTERS = "-- load: counters.csv -> migration_counters"

PREPOPULATE = f"""-- Stage the new schema's identifiers as SCHEMA_MIGRATION aliases (§15.6). Serving under the
-- old schema is untouched; a registry client scoped to the new schema resolves through these.
CREATE TEMP TABLE migration_map (doc_uid UUID PRIMARY KEY, new_sid BYTEA NOT NULL) ON COMMIT DROP;
{LOAD_MAP}
INSERT INTO identifier_aliases (old_semantic_id, id_schema_version, doc_uid, reason, created_at, expires_at)
SELECT new_sid, %(schema)s, doc_uid, 'SCHEMA_MIGRATION', now(), now() + interval '365 days' FROM migration_map
ON CONFLICT (id_schema_version, old_semantic_id) DO UPDATE SET doc_uid = EXCLUDED.doc_uid, expires_at = EXCLUDED.expires_at;
"""

CUTOVER = f"""-- Cutover (§15.6). Rows switch schema; every old identifier becomes a 30-day alias so the old
-- bundle keeps resolving during the rollback window; staged aliases for the new schema go away
-- because the identifiers are now native; counters resume past every planned ordinal.
LOCK TABLE documents IN SHARE ROW EXCLUSIVE MODE;
CREATE TEMP TABLE migration_map (doc_uid UUID PRIMARY KEY, new_sid BYTEA NOT NULL) ON COMMIT DROP;
CREATE TEMP TABLE migration_counters (prefix BYTEA PRIMARY KEY, next_ordinal INTEGER NOT NULL) ON COMMIT DROP;
{LOAD_MAP}
{LOAD_COUNTERS}
INSERT INTO identifier_aliases (old_semantic_id, id_schema_version, doc_uid, reason, created_at, expires_at)
SELECT d.semantic_id, d.id_schema_version, d.doc_uid, 'SCHEMA_MIGRATION', now(), now() + interval '30 days'
FROM documents d JOIN migration_map m USING (doc_uid) WHERE d.id_schema_version <> %(schema)s
ON CONFLICT (id_schema_version, old_semantic_id) DO NOTHING;
UPDATE identifier_history h SET valid_to = now() FROM migration_map m WHERE h.doc_uid = m.doc_uid AND h.valid_to IS NULL;
UPDATE documents d SET semantic_id = m.new_sid, id_schema_version = %(schema)s, updated_at = now()
FROM migration_map m WHERE d.doc_uid = m.doc_uid;
INSERT INTO identifier_history (doc_uid, semantic_id, id_schema_version, valid_from, valid_to)
SELECT doc_uid, new_sid, %(schema)s, now(), NULL FROM migration_map;
DELETE FROM identifier_aliases WHERE id_schema_version = %(schema)s AND reason = 'SCHEMA_MIGRATION';
INSERT INTO prefix_counters (id_schema_version, prefix, next_ordinal)
SELECT %(schema)s, prefix, next_ordinal FROM migration_counters
ON CONFLICT (id_schema_version, prefix) DO UPDATE SET next_ordinal = GREATEST(prefix_counters.next_ordinal, EXCLUDED.next_ordinal);
"""


def plan(corpus: list[dict], embeddings: np.ndarray, ids_cfg: dict, codebook_version: str, out: Path,
         codebooks: str | Path | None = None, sample: int | None = None) -> dict:
    """Assign every live document its new-schema identifier. With ``codebooks`` the frozen
    codebooks are reused (the final re-plan at cutover); otherwise the quantizer is refit."""
    schema = ids_cfg["schema_version"]
    q = RQKMeans.from_config(ids_cfg)
    if codebooks is not None:
        q.codebooks, _ = codebook_io.load(codebooks)
    else:
        n_fit = min(len(corpus), sample or ids_cfg["quantizer"]["sample_size"])
        idx = np.sort(np.random.default_rng(q.seed).choice(len(corpus), n_fit, replace=False))
        q.fit(embeddings[idx])
    codes = q.encode(embeddings)
    order = sorted(range(len(corpus)), key=lambda i: corpus[i]["doc_uid"])  # deterministic ordinals
    counter = PrefixCounter()
    new = {i: counter.allocate(codes[i]) for i in order}

    out.mkdir(parents=True, exist_ok=False)
    cb = codebook_io.save(q.codebooks, out / "codebooks", version=codebook_version, id_schema=schema)
    trie_path, trie_sha = write(new.values(), out / "trie", id_schema=schema, corpus_snapshot=corpus_snapshot_name())
    moved = 0
    with open(out / "mapping.csv", "w", encoding="utf-8", newline="") as f:
        for i in order:
            uid = str(uuid.UUID(corpus[i]["doc_uid"]))  # validated before it reaches the database
            f.write(f"{uid},\\x{new[i].pack().hex()}\n")
            moved += SemanticId.parse(corpus[i]["semantic_id"]).codes[0] != new[i].codes[0]
    with open(out / "counters.csv", "w", encoding="utf-8", newline="") as f:
        for prefix, n in sorted(counter._next.items()):
            f.write(f"\\x{bytes(prefix).hex()},{n}\n")
    (out / "prepopulate.sql").write_text(PREPOPULATE)
    (out / "cutover.sql").write_text(CUTOVER)
    occ = occupancy_report(codes, q.k)
    esc = escape_rate(new.values())
    report = {
        "schema": schema, "codebooks": cb.version, "codebook_sha256": cb.sha256, "refit": codebooks is None,
        "trie": str(trie_path), "trie_sha256": trie_sha, "documents": len(corpus), "escape_rate": esc,
        "level1_moved_share": moved / max(len(corpus), 1), "occupancy": occ,
        "reconstruction_error": q.reconstruction_error(embeddings[: min(len(corpus), 50_000)]),
        "gates": {"escape_rate_ok": esc < ids_cfg["thresholds"]["escape_rate_alarm"],
                  "l1_balance_ok": occ["l1_max_over_mean"] <= ids_cfg["quantizer"]["balance_max_ratio"]},
    }
    (out / "plan.json").write_text(json.dumps(report, indent=2))
    return report


def apply_stage(dsn: str, plan_dir: str | Path, stage: str, approved_by: list[str] | None = None) -> dict:
    """Run one stage in a single transaction. Cutover needs two distinct approvers."""
    plan_dir = Path(plan_dir)
    info = json.loads((plan_dir / "plan.json").read_text())
    if not all(info["gates"].values()):
        raise SystemExit(f"plan failed its own gates, refusing: {info['gates']}")
    if stage == "cutover" and len(set(approved_by or [])) < 2:
        raise SystemExit("cutover requires --approved-by from two different operators")
    import psycopg  # only after every refusal check has passed

    sql = (plan_dir / f"{stage}.sql").read_text()
    loads = {LOAD_MAP: ("migration_map", "mapping.csv"), LOAD_COUNTERS: ("migration_counters", "counters.csv")}
    with psycopg.connect(dsn) as conn, conn.transaction(), conn.cursor() as cur:
        for stmt in _statements(sql):
            if stmt in loads:
                table, name = loads[stmt]
                with cur.copy(f"COPY {table} FROM STDIN WITH (FORMAT csv)") as cp:
                    cp.write((plan_dir / name).read_bytes())
            else:
                cur.execute(stmt, {"schema": info["schema"]})
        cur.execute("INSERT INTO audit_log (actor, op, before_sid, after_sid) VALUES (%s, %s, NULL, %s)",
                    (",".join(sorted(set(approved_by or ["pipeline"]))), f"schema_migration:{stage}", info["schema"]))
    return {"stage": stage, "schema": info["schema"], "documents": info["documents"]}


def _statements(sql: str) -> list[str]:
    out, buf = [], []
    for line in sql.splitlines():
        if line.startswith("-- load:"):
            out.append(line)
        elif not line.startswith("--"):
            buf.append(line)
            if line.rstrip().endswith(";"):
                out.append("\n".join(buf).strip().rstrip(";"))
                buf = []
    return [s for s in out if s]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--corpus", required=True)
    p.add_argument("--embeddings", required=True, help=".npy aligned row-for-row with --corpus")
    p.add_argument("--ids-config", required=True)
    p.add_argument("--codebook-version", required=True)
    p.add_argument("--codebooks", help="reuse these frozen codebooks instead of refitting")
    p.add_argument("--out", required=True)
    a_ = sub.add_parser("apply")
    a_.add_argument("--stage", choices=["prepopulate", "cutover"], required=True)
    a_.add_argument("--plan", required=True)
    a_.add_argument("--dsn", required=True)
    a_.add_argument("--approved-by", action="append", default=[])
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        corpus = [json.loads(x) for x in Path(a.corpus).read_text(encoding="utf-8").splitlines() if x.strip()]
        report = plan(corpus, np.load(a.embeddings), load_yaml(a.ids_config), a.codebook_version, Path(a.out),
                      codebooks=a.codebooks)
        print(json.dumps({k: v for k, v in report.items() if k != "occupancy"}, indent=2))
    else:
        print(json.dumps(apply_stage(a.dsn, a.plan, a.stage, a.approved_by), indent=2))


if __name__ == "__main__":
    main()
