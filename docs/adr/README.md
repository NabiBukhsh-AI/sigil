# Architecture Decision Records

One record per `[FIXED]` decision in the design baseline. A `[FIXED]` decision is
architecturally load-bearing: changing it invalidates other sections and requires a design
review.

Each record states the decision, the alternatives that were considered and rejected, the
consequences accepted alongside it, and — the part that matters most — **what would have to
be true for the decision to be revisited**.

| ADR | Decision | Baseline |
|---|---|---|
| [0001](0001-rq-semantic-identifiers.md) | Residual-quantization semantic identifiers, `L=4`, `K=256`, plus ESCAPE | §5.3 |
| [0002](0002-terminal-fanout-instead-of-learned-ordinal.md) | Terminal fan-out instead of a learned ordinal | §10.4 |
| [0003](0003-trie-colocated-with-decoder.md) | The trie lives in the decoder's process | §4.4, §20.1 |
| [0004](0004-frozen-quantizer-as-compatibility-contract.md) | The frozen quantizer is a compatibility contract | §15.2, §17 |
| [0005](0005-bounded-lexical-channel.md) | A bounded lexical channel, not a dense fallback | §14, §16 |
| [0006](0006-no-ann-in-the-serving-path.md) | No ANN index in the serving path | §2.1 G1, §9.2 |
