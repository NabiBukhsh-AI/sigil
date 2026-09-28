# SIGIL

**Semantic Identifier Generative Index Layer** — a generative retrieval (DSI-class) system that maps a query directly to document identifiers by constrained autoregressive generation, with no approximate-nearest-neighbour index anywhere in the serving path.

```
query ──► encoder ──► constrained beam decode ──► c1.c2.c3.c4 ──► trie fan-out ──► u
                            ▲                                          │
                            └──── trie mask, in-process, mmap ─────────┘
                                                                       ▼
                              registry resolve ──► verify ──► cross-encoder rerank ──► top-k
```

## What this is

Every document is addressed by a residual-quantization semantic code `(c1, c2, c3, c4, u)`
where `c1..c4` are learned quantization codes over a document embedding and `u` is a
disambiguation ordinal. The model decodes the four routing levels autoregressively under a
hard trie constraint; the terminal level is resolved by enumerating the trie's children and
letting a small cross-encoder decide.

The scoring function that decides which documents are retrieved lives in the parameters of a
seq2seq model, not in a distance metric over a stored vector collection.

## Why the shape is what it is

Three structural decisions carry the design:

1. **The model routes; the data structure resolves.** Asking a 220M-parameter model to
   memorize an arbitrary ordinal that distinguishes documents inside one fine-grained
   semantic cell is asking it to do the thing it is worst at — and for a document added after
   the last training run it is not merely hard, it is impossible. So the decoder emits four
   levels, and level five comes from trie fan-out plus reranking. This is what makes
   cold-start retrieval work at all.
2. **The quantizer is frozen and versioned.** A new document's codes are a deterministic
   function of its content, so it lands in the right region of identifier space with no model
   update. Registry and trie updates are therefore independent of model updates.
3. **The trie lives in the decoder's process.** A per-token network hop at beam 64 across
   5 steps would add 320 round trips per query. The constraint is a memory-mapped bitmask
   lookup, not an RPC.

An identifier that does not exist cannot be produced. At every step the logit mask is derived
from the trie, whose sole source of truth is the registry snapshot. `valid_id_rate` is
exactly `1.0` by construction, and any other value is a P1 bug.

## Honest positioning

This architecture is **not** expected to beat a well-built hybrid (BM25 + dense + cross-encoder)
on raw retrieval quality. The strongest published generative retrieval results on large-scale
benchmarks sit around the level of earlier-generation dense retrievers.

It is worth building when the serving index footprint is a binding constraint (a trie over 10M
documents costs 120–200 MB resident against roughly 15 GB for a flat fp16 vector index), when
retrieval must eventually be fused into an LLM's decoding loop, or when a production-grade
generative retrieval implementation is itself the goal.

**Phase 0 exists to let the hybrid baseline win before any model is trained**, and it carries an
explicit stop condition. See `pipelines/` and `libs/sigil_eval/baselines/`.

### Do not use this for

News, social, logs, or any corpus where documents must be retrievable seconds after creation;
corpora above ~50M documents without domain sharding; exact-match, identifier, or code-symbol
lookup (BM25 wins these outright); or teams without GPU capacity or an existing evaluation set.
SIGIL cannot be operated blind.

## Layout

| Path | Contents |
|---|---|
| `libs/sigil_core` | `SemanticId` packing and parsing, bundle manifests, errors, telemetry |
| `libs/sigil_identifiers` | Balanced RQ-KMeans quantizer, codebook I/O, ordinal assignment, ESCAPE |
| `libs/sigil_trie` | Packed on-disk format, compiler, mmap reader, optional Rust extension |
| `libs/sigil_model` | Encoder-decoder with an identifier-only decoder vocabulary and per-level heads |
| `libs/sigil_decoding` | Constrained beam search, terminal fan-out, prefix diversity, confidence |
| `libs/sigil_data` | Ingestion, synthetic query generation, negative mining, dataset mixing |
| `libs/sigil_training` | `L_seq`, `L_rank`, SAM, stratified replay, the four training stages |
| `libs/sigil_eval` | Metrics, baselines, brute-force oracle, and `gates.py` |
| `services/` | Gateway, generative retrieval, reranker, registry, ingestion, trie builder, lexical |
| `pipelines/` | Full train, adapter refresh, dataset build, trie snapshot, schema migration |
| `tests/architecture/` | Parses the serving import graph and fails if a vector index library appears |

`libs/sigil_eval/gates.py` is the single source of truth for release gates. CI, the training
pipeline, and the deployment workflow all import it. Gates defined in three places are gates
enforced in zero.

## Quick start

```bash
make install          # uv sync --all-extras
make test             # unit, property, and architecture tests. No external services needed.
make arch-test        # the no-ANN-in-serving guard on its own
make native           # optional: build the Rust trie extension
```

Nothing above requires a GPU, Postgres, or Redis. The identifier, trie, decoding, and metric
layers are pure and independently testable, which is deliberate — they hold the invariants.

## Operating it

The one question engineers actually ask is *"why did I not get this document?"*, and the answer
is always one of a closed set: pruned at level `l`; retrieved but reranked below `k`; filtered
by ACL or tombstone; not in the trie snapshot; in cold-start state and outranked; or genuinely
low relevance.

```bash
make explain QUERY="canine post op antibiotics" DOC=6f1e2c44
```

Making that set explicit is what turns generative retrieval from a black box into an operable
system.

## Configuration markers

The architecture annotates its own decisions, and the codebase preserves the markers:

- `[FIXED]` — architecturally load-bearing. Changing it invalidates other sections and
  requires a design review. These appear as assertions and architecture tests, not comments.
- `[TUNE: start=X, space=[a,b], gate=<metric>]` — cannot be settled without benchmarking. The
  starting value ships in `configs/`; the named metric is wired into the evaluation harness.
- `[HONEST]` — the architecture does not fully solve this and the residual risk is accepted
  deliberately. Not a TODO.

## Documentation

`docs/architecture.md` points at the design baseline. `docs/adr/` records one decision per
`[FIXED]` choice. `docs/runbooks/` covers rollback, trie rebuild, fallback-creep incidents, and
schema migration — which is the most dangerous operation in the system.

## License

Apache-2.0. See `LICENSE`.
