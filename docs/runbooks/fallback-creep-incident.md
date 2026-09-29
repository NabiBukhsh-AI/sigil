# Runbook: fallback creep

**Baseline:** §14.3, §24 F18, ADR 0005.

Fallback creep is the generative channel quietly losing traffic to lexical fallback until SIGIL
is BM25 with expensive decoration. It is detectable and it is a release blocker, but only if
someone treats the alert as a quality incident rather than a capacity one.

## Signals

| Metric | Alert | Page |
|---|---|---|
| `channel_c_query_share`, rolling 24 h (`GET /v1/ready` on the gateway) | above 15% | above 25% |
| `sigil_results_from_channel{channel="generative"}` share of returned docs at k=10 | below 70% | |

## The one rule

**Do not fix this by touching the fallback.** No raising caps, no suppressing the alert. A
change that improves NDCG mainly by increasing channel C share is rejected by definition. The
cause is in channel A.

## Diagnose by trigger reason

Every channel C invocation is logged with a reason code (`sigil_fallback_trigger_total{reason}`,
and `channels.full_lexical.reasons` in each response). The dominant reason says where to look.

| Reason | What it means | Where to look |
|---|---|---|
| `low_confidence` / `low_margin` | Calibrated confidence under `tau_conf`, or level-1 margin under `tau_margin` | **Calibration drift first.** A new bundle whose stage D failed or was skipped runs uncalibrated, and an uncalibrated model trips this on most queries. Check `calibration.json` exists and ECE in the bundle's `eval_report.json`. If calibration is fine, the model is genuinely unsure: query distribution drift (§24 F8) |
| `verification_dropped_majority` | Tier 1 removed most generative candidates | `trie_registry_skew`: a stale trie ([trie-rebuild.md](trie-rebuild.md)). Or a burst of deletions, or an ACL change hitting one tenant |
| `fewer_than_k_distinct` | Not enough distinct documents survived | Prefix collapse (§24 F11): `sigil_distinct_l1_codes_at_k` falling. Check the diversity quota and the beam's `expansion` |
| `identifier_like_query` | Routing, not failure | The traffic mix changed (a new client sending IDs or code). This is correct behaviour, but it inflates the share: report it separately |
| `gpu_breaker_open` / `generative_error` | The generative service is failing or timing out | Capacity or a bad deploy (§24 F13). Every such response says `degraded_mode=true` |
| `low_relevance` | Top reranked result under `tau_relevant` | Reranker drift, stale snippets, or a model that routes to the wrong region |

## Common fixes

- Stale trie: rebuild ([trie-rebuild.md](trie-rebuild.md)).
- Calibration: re-run stage D on fresh held-out data and ship a new bundle.
- Distribution drift: mine new query logs, weight them 3x, run an adapter refresh; if the refresh
  gate rejects it, schedule the full retrain rather than lowering the gate.
- A bad bundle: [rollback.md](rollback.md).

## Close out

Report both numbers in the weekly quality review: channel C query share and channel A result
share, first, as §14.3 asks.
