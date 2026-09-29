# Runbook: identifier schema migration (codebook refit)

**The single most dangerous operation in the system.**
**Baseline:** §15.4, §15.6, §17.3, §35B R4, ADR 0004.

Every identifier, trie, dataset, and checkpoint depends on the frozen codebooks. Changing them
(new `K`, `L`, embedding encoder, or a refit because of drift or ESCAPE explosion) re-identifies
the whole corpus and needs a model trained on the new identifiers. Budget days, double storage,
and double GPU capacity for the transition. Rehearse on a full-size corpus copy in staging
first, every time.

Two-person rule applies to every step marked **[2P]**.

## How the transition stays safe

Both schemas are live at once. The registry answers the new schema's identifiers through
`SCHEMA_MIGRATION` aliases *before* cutover, and the old schema's through the same aliases
*after* it. Every registry lookup is scoped to one schema, so a bundle only ever resolves
identifiers of its own schema; tier-1 verification accepts an alias hit because the lookup
that produced it was already schema-scoped. A model trained on `ids_v1` still never reads an
`ids_v2` trie (§17.3): tries and bundles stay single-schema; only the registry is dual.

## 1. Plan

```bash
python pipelines/schema_migration.py plan --corpus export/corpus.jsonl --embeddings export/embeddings.npy \
    --ids-config configs/identifiers/ids_v2.yaml --codebook-version cb_v2.0 --out migrations/ids_v2/plan
```

Review `plan.json` before anything else:

- `gates.escape_rate_ok` and `gates.l1_balance_ok` must both be true. If the refit cannot hold
  ESCAPE under 0.5% or level-1 balance under 3x the mean, the new schema is not better than
  the old one. Stop.
- `level1_moved_share` is typically near 1.0: nearly every document moves. That is expected.

## 2. Stage the dual identifier set  **[2P]**

```bash
python pipelines/schema_migration.py apply --stage prepopulate --plan migrations/ids_v2/plan --dsn "$SIGIL_REGISTRY_DSN"
```

This inserts an `ids_v2` alias for every document, in one transaction, with the mapping
bulk-loaded by `COPY`. Nothing that serves `ids_v1` changes. The exact statements are in
`prepopulate.sql` in the plan directory: review them before running. The tool refuses to run a
plan that failed its own gates.

## 3. Train the new bundle

Export the corpus with the planned `ids_v2` identifiers (the `new` column of `mapping.jsonl`),
build the dataset, and train against the planned trie and codebooks:

```bash
python pipelines/dataset_build.py --corpus export/corpus_ids_v2.jsonl ... --version ds_v1_ids_v2
python pipelines/full_train.py --trie migrations/ids_v2/plan/trie/<trie>.trie \
    --codebooks migrations/ids_v2/plan/codebooks ... --bundle-id bundle_ids_v2_a
```

## 4. Shadow

Run the `ids_v2` bundle in the shadow fleet with a registry client scoped to `ids_v2`; it
resolves through the staged aliases. Compare overlap, NDCG, and latency against production
(§15.6). If the shadow comparison fails, discard the bundle; nothing in production changed.

## 5. Cutover window  **[2P]**

1. Pause generative-channel ingestion. New documents still flow to the hot lexical channel,
   so they stay retrievable (ADR 0005).
2. Re-plan with the **frozen** `ids_v2` codebooks so documents added since step 1 are included
   (no refit: `plan ... --codebooks migrations/ids_v2/plan/codebooks --out migrations/ids_v2/final`).
3. Apply the cutover: rows switch to `ids_v2`, every `ids_v1` identifier becomes a 30-day
   `SCHEMA_MIGRATION` alias, staged `ids_v2` aliases are removed, `ids_v2` counters resume past
   every planned ordinal. One transaction, `documents` locked against concurrent writes.
   ```bash
   python pipelines/schema_migration.py apply --stage cutover --plan migrations/ids_v2/final        --dsn "$SIGIL_REGISTRY_DSN" --approved-by alice --approved-by bob
   ```
   The tool refuses without two distinct approvers and records both in the audit log.
4. Stop the `ids_v1` trie builder and pin the last `ids_v1` snapshot. Rebuilding it now would
   produce an empty trie: no rows are `ids_v1` any more.
5. Build and publish the `ids_v2` trie; activate the `ids_v2` bundle (see
   [rollback.md](rollback.md) for the two-person activation calls).
6. Point ingestion at the `ids_v2` codebooks (`SIGIL_CODEBOOKS`) and resume.

## Rolling back

Within the 30-day alias window the `ids_v1` bundle still resolves: its identifiers are aliases
now, and its pinned trie still maps. Documents created after cutover have no `ids_v1`
identifier, so under a rolled-back `ids_v1` bundle they are served by the hot lexical channel
only. After the window, rolling back to `ids_v1` is itself a migration.

## Verify

- `scripts/verify_bundle.py` on the new bundle.
- Golden-set NDCG and `held_out_doc_recall@10` within gates on shadow traffic.
- `sigil_trie_registry_skew_total` flat after the switch.
- `SELECT count(*) FROM documents WHERE id_schema_version = 'ids_v1' AND state <> 'TOMBSTONED'` is 0
  after cutover. Tombstoned rows keep their retired `ids_v1` identifiers.
