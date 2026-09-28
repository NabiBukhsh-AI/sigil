# ADR 0005. A bounded lexical channel, not a dense fallback

- **Status:** Accepted, `[FIXED]`
- **Baseline:** §14, §16, §35B R9
- **Implemented in:** `services/lexical/`, `services/gateway/routing.py`

## Decision

Freshness and tail coverage are handled by a BM25 channel with **hard caps and an alarm that
fires when it starts carrying too much traffic**, so the fallback cannot quietly become the
system.

| Channel | Scope | Cap |
|---|---|---|
| A. Generative | All documents in the trie | Primary |
| B. Hot lexical | Only `ACTIVE_COLD_START` documents | Hot set under 3% of corpus. Alarm at 5%, block ingestion at 10% |
| C. Full lexical | Whole corpus | Under 15% of queries. Page on-call above 25% |

## Why BM25 and not dense ANN

A dense ANN fallback was rejected, and the reason is not cost. **Its failure modes correlate
with the generative model's**, because both derive from the same embedding space. A query
whose prefix gets pruned at level 1 is a query the dense retriever is also likely to
misplace. BM25 fails *independently* of prefix pruning, which is precisely the property a
second channel needs. Adding dense ANN would also reintroduce the exact infrastructure this
architecture exists to avoid.

## The partition is by learning state, not age

"Hot" here means recently changed and not yet learned, deliberately inverted relative to
caching. It matches the actual failure boundary: the generative model's weakness is exactly
and only "documents whose region has not been reinforced by training".

Both channels feed the same reranker, so the score scale is unified and no hand-tuned score
fusion is needed.

## Preventing fallback creep

These are release blockers, enforced by the evaluation gate and by production alerts:

- `channel_c_query_share` under 15 percent over a rolling 24 h window.
- `results_from_channel_a_share` above 70 percent of returned documents at k=10.
- Every response carries `channel_attribution` per document, and the weekly quality review
  reports these two numbers first.
- **If a change improves NDCG mainly by increasing channel C share, it is rejected.** The
  system's reason for existing is channel A.

## Accepted compromises `[HONEST]`

1. A second retrieval system exists, with real correctness and operational burden. Mitigated
   by keeping it a single embedded index (Tantivy), never a distributed search cluster.
2. Ranking is inconsistent across the boundary. A hot and a cold document compete only
   through the reranker, which sees a 256-token snippet. Measured as
   `cross_channel_ordering_error`.
3. It creates an incentive to let the hot set grow. Countered by the hard caps above.
4. It fixes reachability, not ranking, for high-churn corpora. If most queries target
   recently changed documents, most retrieval is effectively BM25 and the generative model is
   decoration. That is the condition under which this architecture should be abandoned.

## The generative-native alternative

A SEAL-style substring channel would keep freshness inside the generative paradigm and has
strong published evidence for the generalization property. Deferred to Phase 12: the
FM-index is a large auxiliary structure with variable-length decoding, and BM25 reaches a
similar practical outcome at a fraction of the engineering cost. If purity of paradigm
matters more than delivery speed, swap Channel B for the substring channel and accept
roughly one additional engineer-quarter.
