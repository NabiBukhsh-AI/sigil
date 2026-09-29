# Runbook: trie rebuild

**Baseline:** §12.4, §18.3, §24 F2 and F16, ADR 0003.

The trie is a derived artefact of the registry. It is never edited, only rebuilt, and a rebuild
is always safe: it reads the registry, validates the result against the same id set, and only
then publishes.

## When

- `sigil_trie_registry_skew_total` rising, or skew above 0.1 percent of candidates (paging alert).
  The trie holds identifiers the registry no longer knows, or is missing new ones.
- `sigil_trie_snapshot_age_seconds` well past the 15-minute cadence: the builder is stuck.
- A pod reports it refused a snapshot (`/snapshot` answered 422): hash mismatch on load (F16).
- After restoring the registry from backup.

## Rebuild once

```bash
python -m services.trie_builder.main --config configs/serving/prod.yaml --out $TRIE_DIR --once
```

It prints the snapshot info, including `n_ids`, `ids_sha256`, `build_seconds`, and the per-pod
acknowledgements. A 10M-document build should finish in under 2 minutes (Phase 3 acceptance);
`benchmarks/scale` measures it on synthetic corpora.

Validation compares the compiled trie against the registry's live identifier set before anything
is published: exactly those paths, nothing else. A snapshot that fails validation is deleted,
never published, and the command exits non-zero.

## If pods refuse the new snapshot

Pods verify SHA-256 before mapping and keep the previous snapshot if verification fails, so a
refusal is safe but stale. Check, in order:

1. The file on shared storage matches its `.sha256` sidecar (`sha256sum`). A mismatch is a
   storage fault or a truncated copy: rebuild to a fresh path.
2. The trie's `id_schema` equals the bundle's. A mismatch is a deployment error, not a trie
   problem; see [schema-migration.md](schema-migration.md).
3. The pod can read the path (mount, permissions).

## If the builder keeps exceeding its window

Lengthen the cadence (`trie.snapshot_minutes`) and alert; do not skip validation to go faster.
New documents are served by the hot lexical channel meanwhile (ADR 0005), so a slower trie
cadence delays generative reachability, not retrievability.

## Verify

```bash
curl -s $GRS/health | jq '.trie_snapshot, .trie_sha256'      # every pod on the same snapshot
```

The gateway refuses to mix pods on different snapshots within one request, so a partial rollout
shows up as uneven pod health, not as inconsistent results.
