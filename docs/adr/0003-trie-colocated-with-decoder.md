# ADR 0003. The trie lives in the decoder's process

- **Status:** Accepted, `[FIXED]`
- **Baseline:** §4.4, §20.1
- **Implemented in:** `libs/sigil_trie/mmap_reader.py`, `services/generative_retrieval/`

## Decision

The trie mask provider is a memory-mapped structure held **in the same process** as the
decoder. It must not be behind an RPC.

## Reasoning

Boundaries in this system are drawn by **coupling frequency**: anything touched per decode
step lives in the same process; anything touched per request may cross a network boundary;
anything touched per deployment is a separate system.

`children()` is called once per beam per step. At beam 64 across 5 steps that is 320 lookups
per query. Behind an RPC those become 320 network round trips, which dominates every other
cost in the latency budget by an order of magnitude.

This is why the strawman decomposition separating a "Decoder Service" from a "Constrained
Decoding Service" was removed: it would have placed a network hop inside the token loop.

## Consequences

- Every generative replica holds a full copy of the trie, roughly 120–200 MB at 10M
  documents, on top of ~500 MB of weights. Replication cost is acceptable precisely because
  the trie is two orders of magnitude smaller than the vector index it replaces.
- Snapshots are immutable, content-addressed, and hash-verified on both write and load. A
  swap is an atomic pointer flip after the new file is fully mapped and verified, so no
  request in flight is affected.
- The generative service is stateful *in the loading sense*. It reports its active snapshot
  version in the health endpoint, and the gateway refuses to mix results from pods on
  different snapshots within one request.
- Per-step masking in pure Python would add several milliseconds per query, so
  `libs/sigil_trie/native/` is a Rust extension. This is the one place where a native
  extension is justified on measurement rather than preference. A pure-Python reader ships
  alongside it as the reference implementation, and a fuzz test asserts they agree.

## What would revisit this

Nothing short of a decode architecture where masking is no longer per-step.
