# ADR 0004. The frozen quantizer is a compatibility contract

- **Status:** Accepted, `[FIXED]`
- **Baseline:** §15.2, §17, §35B R4
- **Implemented in:** `libs/sigil_core/bundle.py`, `libs/sigil_identifiers/quantizer/codebook_io.py`

## Decision

Codebooks are frozen, versioned, content-hashed artefacts. They are the only global coupling
in the design and are treated with the ceremony that implies. Every bundle manifest names a
`CodebookVersion` and its SHA-256, and compatibility is asserted at load time.

## Why the freeze buys so much

The frozen quantizer is what makes the registry and trie layers independent of the model
layer. A new document's codes are a deterministic function of its embedding, so it is placed
correctly in identifier space without any model update:

| Layer | Update cost | Update latency |
|---|---|---|
| Registry (data) | O(1) insert | under 1 s |
| Trie + codes (data, derived) | O(1) delta, O(N) snapshot | under 15 min |
| Model (parameters) | Adapter training | under 24 h |

What the model does not yet know is that a *query* should route to that region more strongly
than before. That is a ranking degradation, not an unreachability, and that distinction is
the whole reason this architecture is deployable.

## The risk this creates

Every identifier, every trie, every dataset, and every checkpoint depends on the codebooks.
Changing them is a full corpus reindex plus full retrain plus dual-bundle migration, measured
in days and in doubled infrastructure. It is the single most dangerous operation in the
system.

## Controls

- Codebooks are first-class versioned artefacts with hashes in the bundle manifest.
- Load-time assertions: `codebooks.id_schema == bundle.id_schema`, `trie.id_schema ==
  bundle.id_schema`, `adapter.backbone == bundle.backbone`. A pod that fails any of these
  refuses to become ready rather than serving degraded results.
- A model trained on `ids_v2` may never read an `ids_v3` trie. Different token semantics
  entirely, so this is a hard failure and never a graceful degradation.
- `docs/runbooks/schema-migration.md` is rehearsed in staging on a full-size corpus copy.
- Codebook refit is on the **quarterly full-retrain cadence as an evaluated option**, not
  reserved for emergencies, so the migration path is exercised routinely instead of first
  being attempted under pressure.

## What would revisit this

An incremental quantizer (CLEVER-style incremental product quantization) that permits
codebook extension without permuting existing identifiers. Phase 12 research item.
