# Runbook: bundle rollback

**Goal:** return serving to the previous bundle in under 5 minutes (G8), with no registry change.
**Baseline:** §17.4, §18.5, §36.6.

Rollback is a pointer flip. The registry is shared and never versioned with the bundle, which is
what makes this safe: the rows the previous bundle expects are still there.

## When

Any of: golden-set NDCG proxy down more than 5 percent; `sigil_valid_id_rate` not exactly 1.0;
p95 above 250 ms for 10 minutes after a deploy; an adapter that passed its gates but misbehaves
online; operator judgment during a canary. Roll back first, investigate second.

## Before you flip: one check

A rollback **must not cross an identifier-schema migration**. If the previous bundle's
`id_schema` differs from the one the registry rows now carry, rolling back makes every
generative candidate fail tier-1 schema verification. That case is
[schema-migration.md](schema-migration.md), section *Rolling back*.

```bash
curl -s $VM/v1/models:active                       # active + the 3 hot-loadable bundles
curl -s $VM/v1/models/<previous_bundle_id> | jq .id_schema
curl -s $REGISTRY/internal/epoch                    # sanity: registry reachable
```

If the previous bundle is not in `hot`, it is in object storage but not warm; budget the extra
pod-start time (model load plus trie mmap, under 5 s for the trie).

## Flip (two-person rule)

```bash
# operator A
curl -s -X POST $VM/v1/models:activate -H "Authorization: Bearer $OP_A" -d '{"bundle_id": "<previous>"}'
# -> {"status": "pending", "approvers": ["a"]}
# operator B
curl -s -X POST $VM/v1/models:activate -H "Authorization: Bearer $OP_B" -d '{"bundle_id": "<previous>"}'
# -> {"status": "active", ...}
```

Then roll the generative and gateway deployments onto the previous bundle (Helm sets
`SIGIL_BUNDLE` / `SIGIL_BUNDLE_MANIFEST` from the active pointer):

```bash
helm upgrade sigil deployment/helm/sigil --reuse-values --set bundle.id=<previous>
```

Pods verify every hash in the manifest before reporting ready (§17.3). A pod that cannot verify
never takes traffic, so a bad artefact cannot turn a rollback into an outage.

## Verify

```bash
for p in $(kubectl get pods -l app=sigil-generative -o name); do kubectl exec $p -- curl -s localhost:8081/health; done
curl -s $GW/v1/retrieve -H "Authorization: Bearer $KEY" -d '{"query":"<golden query>","tenant_id":"<t>"}' | jq .bundle_id
python scripts/verify_bundle.py --bundle <path-to-previous-bundle>
```

Every response is stamped with `bundle_id`; confirm it before declaring done. Record the
elapsed time: §34 wants the drill timed.

## After

Freeze deploys. Open the incident. Diff the two bundles' `eval_report.json`; the answer to
"what changed" is almost always the dataset version or the adapter, both named in the manifest.
