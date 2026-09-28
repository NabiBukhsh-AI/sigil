# ADR 0001. Residual-quantization semantic identifiers

- **Status:** Accepted, `[FIXED]`
- **Baseline:** §5.3
- **Implemented in:** `libs/sigil_core/ids.py`, `libs/sigil_identifiers/quantizer/`

## Decision

Documents are addressed by residual-quantization semantic codes, `L=4` levels of `K=256`,
plus a disambiguation level `u`, with an ESCAPE mechanism for overflow.

```
docid := c1 . c2 . c3 . c4 . u      c_i in [0,255],  u in [0,254]
         u = 255 is ESCAPE, meaning "one more level follows"
```

## Alternatives considered

**Atomic IDs** — one vocabulary token per document. Often the strongest option at small
`N`, and rejected here for concrete reasons: the output embedding matrix is `N x 768`,
which at 10M documents is 7.7B parameters in the softmax alone; softmax cost is linear in
`N`; every new document adds a randomly initialized row that has never received gradient;
and the identifiers carry zero semantic structure, so nothing generalizes across documents.

**Human-readable taxonomy paths** — excellent for interpretability, filtering, and
debugging. Requires a maintained taxonomy, produces heavily imbalanced branches (the "misc"
leaf swallows 40 percent of a real corpus), and taxonomy revisions are a full reindex.
Retained as *metadata*, not as the identifier.

**Hierarchical k-means codes** — DSI's original semantic identifier. Balanced by
construction, but the tree must be rebuilt when the corpus distribution shifts, and
rebuilding permutes every identifier.

**Distinctive n-gram identifiers (SEAL-style)** — the one family that generalizes to unseen
documents without retraining, which is a genuinely decisive property. Costs a large
auxiliary FM-index, variable-length decoding, weak performance on paraphrastic queries, and
much messier score aggregation. Deferred to Phase 12; see ADR 0005.

## Why RQ codes

- A bounded 1280-token vocabulary decouples model size from corpus size, which is the
  property atomic IDs fatally lack.
- The coarse-to-fine residual structure means beam pruning removes a semantically coherent
  region rather than an arbitrary one, and the model's first two decisions become genuinely
  learnable generalizations rather than memorization.
- Frozen codebooks make identifier assignment for a new document a pure function of its
  content, so insertion needs no model update to place the document correctly.
- It is the only identifier family with published evidence of working at the 8.8M-passage
  scale (RIPOR, WWW 2024).

## Consequences

Levels 1 and 2 are fit with **balanced** k-means (capacity-constrained assignment) so no
coarse partition exceeds 3x the mean occupancy. If a 4-level prefix accumulates more than
255 documents, the 256th and later take `u=255 (ESCAPE)` followed by an additional byte
level. An escape rate above 0.5 percent is a signal that the quantizer needs refitting.

A document's identifier changes only if its content moves it across a quantization
boundary. Metadata updates never change it. When it does change, the old identifier becomes
an alias, never a dangling path.

## What would revisit this

`escape_rate` persistently above 0.5 percent with a well-fit quantizer; evidence that
level-1 occupancy entropy collapses on the target corpus; or Experiment 3 showing
hierarchical k-means beating RQ on `prefix_survival@2`.
