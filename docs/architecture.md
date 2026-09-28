# Architecture

The design baseline for SIGIL (`ARCH-1.0`) is maintained in the private companion
repository, `NabiBukhsh-AI/sigil-internal`, as
`SIGIL_generative_retrieval_architecture.md`.

It is not mirrored here. This file is a pointer and a map, so that section references
scattered through the codebase resolve to something.

## Section map

Code in this repository cites the baseline by section number. The mapping:

| Section | Subject | Implemented in |
|---|---|---|
| §4 | Core retrieval mechanism, confidence estimation | `libs/sigil_decoding/` |
| §5 | Document identifier design, RQ codes, ESCAPE | `libs/sigil_core/ids.py`, `libs/sigil_identifiers/` |
| §6 | Model architecture, per-level heads, identifier vocabulary | `libs/sigil_model/` |
| §7 | Training data construction, synthetic queries, coverage floor | `libs/sigil_data/` |
| §8 | Training objectives `L_seq`, `L_rank`, calibration | `libs/sigil_training/objectives.py`, `stages.py` |
| §9 | Negative sampling, false-negative filtering | `libs/sigil_data/negatives/` |
| §10 | Constrained beam search, terminal fan-out, pruning | `libs/sigil_decoding/beam.py`, `fanout.py` |
| §11 | Multi-document retrieval, prefix diversity quotas | `libs/sigil_decoding/diversity.py` |
| §12 | Document resolution, registry schema, aliases | `services/registry/`, `libs/sigil_registry_client/` |
| §13 | Tier-1 verification, reranking, score blending | `services/gateway/verification.py`, `services/reranker/` |
| §14 | Hybrid channels and fallback caps | `services/gateway/routing.py`, `services/lexical/` |
| §15 | Corpus update, adapter refresh, replay | `libs/sigil_training/replay.py`, `pipelines/adapter_refresh.py` |
| §17 | Versioning, bundle manifests, compatibility rules | `libs/sigil_core/bundle.py`, `services/version_manager/` |
| §19 | Storage architecture | `deployment/` |
| §21 | API design | `services/gateway/schemas.py` |
| §22 | Latency budget and optimizations | `benchmarks/latency/` |
| §25 | Evaluation framework, metrics, baselines | `libs/sigil_eval/` |
| §26 | Experimental framework E0–E12 | `libs/sigil_eval/`, `scripts/simulate_churn.py` |
| §28 | Observability, the closed-set diagnostic | `libs/sigil_core/telemetry.py`, `scripts/explain_query.py` |
| §31 | Project structure | this repository's layout |
| §32 | Implementation phases | `docs/adr/`, commit history |
| §33 | Testing strategy | `tests/` |

## Decision markers

The baseline annotates its own decisions and the codebase preserves the markers verbatim:

- **`[FIXED]`** — load-bearing. Changing it invalidates other sections and requires a
  design review. Where possible these are enforced mechanically (load-time assertions,
  `tests/architecture/`) rather than left as comments.
- **`[TUNE: start=X, space=[a,b], gate=<metric>]`** — cannot be settled without
  benchmarking. The starting value ships in `configs/`, and the named metric is wired
  into `libs/sigil_eval/`.
- **`[HONEST]`** — the architecture does not fully solve the problem and the residual
  risk is accepted deliberately. These are not TODOs and should not be "fixed" without
  revisiting the baseline.

## Architecture decision records

`docs/adr/` carries one record per `[FIXED]` decision, stating the decision, the
alternatives considered, and what would have to change for the decision to be revisited.
