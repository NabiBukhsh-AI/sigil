# ADR 0002. Terminal fan-out instead of a learned ordinal

- **Status:** Accepted, `[FIXED]`
- **Baseline:** §10.4, §35B R1
- **Implemented in:** `libs/sigil_decoding/fanout.py`

## Decision

The decoder generates only the four semantic routing levels autoregressively. The terminal
level `u` is resolved by enumerating the trie's children under each surviving 4-level
prefix and letting the cross-encoder reranker decide.

## Context

This is the change forced by the sharpest finding of the adversarial design review.

A 220M-parameter model over 10M documents has roughly **22 parameters per document**.
Asking it to emit `u` — an arbitrary ordinal distinguishing documents inside one
fine-grained semantic cell — is asking it to memorize an assignment with no learnable
structure. For a document added after the last training run it is not merely hard, it is
impossible: the model has never seen that ordinal.

## Consequences

**Positive.** Any document whose content places it under a reachable 4-level prefix is
retrievable the moment it enters the trie, with no model update. This is what makes
cold-start retrieval work at all, and Experiment 7 exists specifically to validate it.

**Positive.** The model's capacity is spent entirely on routing, which is the part that
generalizes.

**Negative.** The system is now "learned coarse router plus cheap verifier", not a pure DSI
in which the corpus lives in the parameters. That is a deliberate departure from the
paradigm's purest form.

**Negative.** The reranker stops being optional. It is the only component that can rescue
the fan-out, so its availability is a hard dependency of retrieval quality, and §24 F19
defines the degraded path when it is down.

## What would revisit this

Experiment 7 showing that fan-out does not raise `cold_doc_recall@10` above roughly 0.5 of
steady state. If fan-out does not help, the architecture is wrong and this ADR — not the
reranker — is what needs replacing.
