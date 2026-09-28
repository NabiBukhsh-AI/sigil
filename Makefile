# SIGIL. See docs/architecture.md for the section numbers referenced throughout.
.DEFAULT_GOAL := help
SHELL := /bin/bash

PY      ?= uv run
CONFIG  ?= configs/serving/dev.yaml
CORPUS  ?= 100k

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Sync the uv workspace with dev + training extras
	uv sync --all-extras

.PHONY: native
native: ## Build the Rust trie extension (libs/sigil_trie/native)
	cd libs/sigil_trie/native && cargo build --release
	@echo "Set SIGIL_TRIE_NATIVE=1 to use it. Falls back to the Python reader otherwise."

.PHONY: lint
lint: ## ruff + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy libs services

.PHONY: fmt
fmt: ## Apply ruff formatting
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: test
test: ## Unit, property, and architecture tests. No external services.
	uv run pytest tests/unit tests/architecture tests/retrieval_quality -q

.PHONY: test-all
test-all: ## Everything, including integration and lifecycle
	uv run pytest -q

.PHONY: arch-test
arch-test: ## §9.2 / §31 guard: no ANN library may appear in the serving import graph
	uv run pytest tests/architecture -q

.PHONY: train
train: ## Full training run, stages A to D (§8)
	$(PY) python pipelines/full_train.py --config configs/model/base.yaml

.PHONY: refresh
refresh: ## Nightly LoRA adapter refresh with stratified replay (§15.5)
	$(PY) python pipelines/adapter_refresh.py --config configs/model/base.yaml

.PHONY: dataset
dataset: ## Build a versioned training dataset (§7)
	$(PY) python pipelines/dataset_build.py --config configs/model/base.yaml

.PHONY: trie
trie: ## Compile and publish a trie snapshot from the registry (§18.3)
	$(PY) python pipelines/trie_snapshot.py --config $(CONFIG)

.PHONY: eval
eval: ## Full §25 metric set + all §25.2 baselines + gate verdict
	$(PY) python -m sigil_eval.report --config configs/gates/release_gates.yaml

.PHONY: serve
serve: ## Run the gateway locally against dev config
	$(PY) uvicorn services.gateway.main:app --host 0.0.0.0 --port 8080 --reload

.PHONY: serve-grs
serve-grs: ## Run the generative retrieval service (GPU)
	$(PY) uvicorn services.generative_retrieval.main:app --host 0.0.0.0 --port 8081

.PHONY: bench
bench: ## Latency and throughput benchmarks (§22)
	$(PY) python benchmarks/latency/run.py
	$(PY) python benchmarks/throughput/run.py

.PHONY: bench-scale
bench-scale: ## Experiment 9 scale sweep. CORPUS=10k|100k|1m
	$(PY) python benchmarks/scale/run.py --corpus $(CORPUS)

.PHONY: migrate
migrate: ## Codebook / id-schema migration. The most dangerous operation in the system.
	@echo "Read docs/runbooks/schema-migration.md first. Two-person rule applies."
	$(PY) python pipelines/schema_migration.py --config $(CONFIG) --dry-run

.PHONY: explain
explain: ## Why did I not get this document? QUERY=... DOC=...
	$(PY) python scripts/explain_query.py --query "$(QUERY)" --doc "$(DOC)"

.PHONY: verify-bundle
verify-bundle: ## Check a bundle manifest for §17.3 compatibility violations
	$(PY) python scripts/verify_bundle.py --bundle $(BUNDLE)

.PHONY: churn
churn: ## Replay N days of corpus mutation for Experiment 8
	$(PY) python scripts/simulate_churn.py --days 30
