"""Phase 0 cost model. §32 Phase 0, §23.2, §35B R6, Experiment 10.

    python benchmarks/cost_model.py --docs 1000000 --qps 200 --gpu-hour 1.20

The decision this feeds is build-versus-buy against a hybrid (BM25 + dense + cross-encoder),
so it prices the three things that differ: synthetic query generation (the dominant offline
cost, 60 to 80 percent of the program's offline budget), serving GPU time per query, and
index memory. Every default is an assumption to replace with a measurement: generator
throughput from a real batch job, per-query GPU ms from benchmarks/latency on target hardware,
trie bytes per document from benchmarks/scale.
"""

from __future__ import annotations

import argparse
import json


def model(a: argparse.Namespace) -> dict:
    units = a.docs * a.units_per_doc
    # Offline: synthetic query generation. Input is the unit, output ~a.query_tokens per query.
    gen_tokens = units * a.queries_generated * (a.query_tokens + a.unit_tokens / a.prompt_share)
    gen_gpu_h = gen_tokens / a.gen_tokens_per_s / 3600
    # Offline: training, stages A-C, in tokens processed.
    train_tokens = (a.stage_a_steps + a.stage_bc_steps) * a.batch_tokens
    train_gpu_h = train_tokens / a.train_tokens_per_s / 3600
    # Serving: per-query GPU ms for encode + decode + rerank; the cache removes a share.
    gpu_ms = (a.encode_ms + a.decode_ms + a.rerank_ms) * (1 - a.cache_hit)
    queries_per_gpu_s = 1000 / gpu_ms * a.batching_gain
    gpus = a.qps / queries_per_gpu_s
    serve_per_1k = a.gpu_hour * gpus / (a.qps * 3.6)
    dense_per_1k = a.gpu_hour * (a.qps / (1000 / (a.dense_encode_ms + a.rerank_ms) * a.batching_gain)) / (a.qps * 3.6)
    # Index memory.
    trie_gb = units * a.trie_bytes_per_doc / 2**30
    vec_gb = units * a.dim * 2 / 2**30  # fp16
    hnsw_gb = vec_gb * (1 + a.hnsw_overhead)
    return {
        "units": units,
        "offline": {"synthetic_generation_gpu_hours": round(gen_gpu_h, 1),
                    "synthetic_generation_cost": round(gen_gpu_h * a.gpu_hour, 2),
                    "training_gpu_hours": round(train_gpu_h, 1), "training_cost": round(train_gpu_h * a.gpu_hour, 2),
                    "generation_share_of_offline": round(gen_gpu_h / (gen_gpu_h + train_gpu_h), 3)},
        "serving": {"gpus_needed": round(gpus, 2), "cost_per_1k_queries": round(serve_per_1k, 5),
                    "dense_hybrid_cost_per_1k_queries": round(dense_per_1k, 5),
                    "ratio_vs_dense": round(serve_per_1k / dense_per_1k, 2)},
        "index_memory_gb": {"sigil_trie": round(trie_gb, 3), "fp16_flat_vectors": round(vec_gb, 2),
                            "hnsw": round(hnsw_gb, 2), "ratio_hnsw_over_trie": round(hnsw_gb / trie_gb, 1)},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add = ap.add_argument
    add("--docs", type=int, default=1_000_000)
    add("--units-per-doc", type=float, default=1.0)
    add("--queries-generated", type=int, default=16, help="sampled per unit before filtering to 10 (§7.3)")
    add("--query-tokens", type=int, default=20)
    add("--unit-tokens", type=int, default=512)
    add("--prompt-share", type=float, default=16, help="prompt tokens are shared across the n samples of one unit")
    add("--gen-tokens-per-s", type=float, default=20_000, help="generator throughput per GPU, continuous batching")
    add("--stage-a-steps", type=int, default=300_000)
    add("--stage-bc-steps", type=int, default=80_000)
    add("--batch-tokens", type=int, default=65_000)
    add("--train-tokens-per-s", type=float, default=150_000)
    add("--qps", type=float, default=200)
    add("--encode-ms", type=float, default=5)
    add("--decode-ms", type=float, default=9)
    add("--rerank-ms", type=float, default=13)
    add("--dense-encode-ms", type=float, default=3)
    add("--cache-hit", type=float, default=0.25, help="candidate cache, 15 to 40 percent (§22.3)")
    add("--batching-gain", type=float, default=3.0, help="continuous batching, 2 to 4x (§22.3)")
    add("--gpu-hour", type=float, default=1.20)
    add("--trie-bytes-per-doc", type=float, default=10.0, help="benchmarks/scale measured ~9.6 at 1M")
    add("--dim", type=int, default=768)
    add("--hnsw-overhead", type=float, default=0.45)
    print(json.dumps(model(ap.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
