# ADR 0006. No ANN index in the serving path

- **Status:** Accepted, `[FIXED]`
- **Baseline:** §2.1 G1, §9.2, §19
- **Enforced by:** `tests/architecture/test_no_ann_in_serving.py`

## Decision

No approximate-nearest-neighbour index, and no vector database, appears anywhere in the
serving call graph. This is goal G1 and it is enforced mechanically, not by discipline.

## The tension this resolves

An offline ANN index **is** used, for negative mining (§9.1), where roughly 10 percent of
negatives come from a top-100 embedding lookup. That index is built from a snapshot inside
the training job, is never deployed, and is discarded with the job. It has the same status as
the BM25 index used in the round-trip consistency filter: a data-generation tool.

The distinction matters enough to be worth enforcing, because it is exactly the kind of
boundary that erodes. A future engineer with a latency problem and a FAISS index already in
the repository has an obvious and wrong shortcut available.

## Enforcement

`tests/architecture/` parses the import graph of every module reachable from `services/` and
fails the build if `faiss`, `scann`, `hnswlib`, `annoy`, `usearch`, `pinecone`, `weaviate`,
`qdrant`, `milvus`, `chromadb`, or `lancedb` appears. The test walks transitive first-party
imports, so hiding the dependency one module deeper does not evade it.

CI runs it as a separate `make arch-test` target so the failure is legible: this is an
architecture violation, not a broken unit test.

## What would revisit this

Nothing within this architecture's stated goals. If a deployment concludes it needs ANN in
the serving path, it does not need SIGIL. It needs a hybrid dense system, and §35.3 says so
plainly.
