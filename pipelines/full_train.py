"""Full training run: stages A to D, offline gates, bundle assembly. §8, §17.2, §29.

    python pipelines/full_train.py --config configs/model/base.yaml --dataset datasets/ds_v1 \
        --corpus export/corpus.jsonl --golden-queries golden/queries.tsv --golden-qrels golden/qrels.txt \
        --trie artifacts/tries/trie_<sha>.trie --codebooks artifacts/codebooks/cb_v1.0 \
        --bundle-id bundle_2026_09_02_a --out artifacts/bundles

[FIXED] Nothing is deployed except a complete, signed bundle. A run that fails its offline
gates writes its report and no bundle. The release decision proper happens after shadow
traffic, where the remaining gates (channel shares, latency, cost, tenant leaks) are measured.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from sigil_core.bundle import BundleManifest
from sigil_core.config import load_yaml
from sigil_core.ids import SemanticId
from sigil_decoding import DecodeConfig, decode
from sigil_decoding.confidence import Calibrator
from sigil_eval import gates, metrics


def evaluate_scorer(scorer, trie, golden: Sequence[tuple[str, set[str]]], sid_to_doc: dict[str, str],
                    cfg: DecodeConfig, calibrator: Calibrator = Calibrator(), held_out: set[str] = frozenset()) -> dict:
    """Generative-channel offline metrics on (query, relevant doc_uids) pairs."""
    run, qrels, emitted, probs, labels, held_q = {}, {}, [], [], [], []
    for n, (q, rel) in enumerate(golden):
        qid = f"q{n}"
        r = decode(scorer, q, trie, cfg, calibrator)
        emitted += [c.sid for c in r.candidates]
        run[qid] = [sid_to_doc[str(c.sid)] for c in r.candidates if str(c.sid) in sid_to_doc]
        qrels[qid] = dict.fromkeys(rel, 1)
        if r.candidates:
            probs.append(r.confidence)
            labels.append(float(bool(run[qid]) and run[qid][0] in rel))
        if rel and rel <= held_out:
            held_q.append(qid)
    out = metrics.evaluate_run(run, qrels)
    out["valid_id_rate"] = metrics.valid_id_rate(emitted, trie)
    out["ece"] = metrics.ece(np.array(probs), np.array(labels)) if probs else 1.0
    out["held_out_doc_recall@10"] = (metrics.evaluate_run(run, qrels, queries=held_q)["recall@10"]
                                     if held_q else None)
    out["stale_id_rate"] = 0.0  # offline: the trie and the registry export are one snapshot
    return out


def loop_config(stage: dict, **over):
    from sigil_training.loop import LoopConfig

    return LoopConfig(
        steps=over.pop("steps", stage.get("steps_per_round", stage.get("steps", 1000))),
        lr=stage.get("peak_lr_heads", stage.get("lr", 1e-5)), lr_backbone=stage.get("peak_lr_backbone", stage.get("lr", 1e-5)),
        warmup=stage.get("warmup_steps", 500), schedule="constant" if stage.get("lr_schedule") == "constant" else "inverse_sqrt",
        label_smoothing=stage.get("label_smoothing", 0.1), level_weights=stage.get("level_weights", (1, 1, 1, 1, 0.5)),
        lambda_rank=stage.get("lambda_rank", 0.0), margins=stage.get("margins", (0.5, 0.5, 1, 1, 1)),
        sam=stage.get("sam_enabled", False), sam_rho=stage.get("sam_rho", 0.05), **over,
    )


def train_bundle(*, model, tokenizer, rows: list[dict], cfg: dict, trie_path: str, trie_sha256: str,
                 codebooks_dir: str, corpus: list[dict], golden: list[tuple[str, set[str]]], held_out: set[str],
                 out_dir: str, bundle_id: str, backbone: str, dataset: str, corpus_snapshot: str,
                 epoch_range: tuple[int, int], incumbent: dict | None = None, signing_key: bytes | None = None,
                 steps: int | None = None, batch_size: int = 256, device: str | None = None,
                 thresholds: dict | None = None) -> dict:
    import torch
    from sigil_data.datasets import batches
    from sigil_identifiers.quantizer import codebook_io
    from sigil_model.encoder_decoder import ModelScorer
    from sigil_trie import TrieSnapshot
    from sigil_training import stages

    st = cfg["training"]["stages"]
    over = {"batch_size": batch_size, **({"steps": steps} if steps else {})}
    decode_cfg = DecodeConfig(beam=st["c_self_negative"]["beam"])
    sid_to_doc = {r["semantic_id"]: r["doc_uid"] for r in corpus}
    # Held-out-document queries are scarce and gate cold-start (§7.5): all of them go to
    # evaluation, never to calibration. The rest split in half.
    held_q = [g for g in golden if g[1] and g[1] <= held_out]
    rest = [g for g in golden if not (g[1] and g[1] <= held_out)]
    calib, test = rest[: len(rest) // 2], rest[len(rest) // 2 :] + held_q
    report: dict = {"bundle_id": bundle_id}

    with TrieSnapshot(trie_path, trie_sha256) as trie:
        report["stage_a"] = stages.stage_a(model, batches(rows, batch_size), tokenizer,
                                           loop_config(st["a_indexing"], **over), device=device)[-1:]
        report["stage_b"] = stages.stage_b(model, batches(rows, batch_size, seed=1), tokenizer,
                                           loop_config(st["b_prefix_rank"], **over), device=device)[-1:]
        scorer = ModelScorer(model, tokenizer, device or "cpu")

        def make_batches(transform, _i=[0]):  # noqa: B006  a fresh stream per round
            _i[0] += 1
            for b in batches(rows, batch_size, seed=10 + _i[0]):
                yield transform(b)

        c = st["c_self_negative"]
        report["stage_c"] = stages.stage_c(model, make_batches, tokenizer, scorer, trie,
                                           loop_config({**st["b_prefix_rank"], **c}, **over), decode_cfg,
                                           rounds=c["rounds"], device=device)[-1:]
        doc_to_sid = {v: SemanticId.parse(k) for k, v in sid_to_doc.items()}
        try:
            cal, report["stage_d"] = stages.stage_d(scorer, trie, [(q, {doc_to_sid[d] for d in rel if d in doc_to_sid})
                                                                    for q, rel in calib], decode_cfg)
        except ValueError as e:  # confidence does not track relevance: uncalibrated, and the ECE gate decides
            cal, report["stage_d"] = Calibrator(), {"error": str(e)}
        candidate = evaluate_scorer(scorer, trie, test, sid_to_doc, decode_cfg, cal, held_out)

    escape = sum(SemanticId.parse(r["semantic_id"]).escaped for r in corpus) / max(len(corpus), 1)
    candidate["escape_rate"] = escape
    verdict = gates.evaluate({k: v for k, v in candidate.items() if v is not None}, incumbent,
                             thresholds=thresholds, only=gates.OFFLINE)
    report.update(metrics=candidate, offline_gates={"passed": verdict.passed, "summary": verdict.summary()})

    out = Path(out_dir) / bundle_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "eval_report.json").write_text(json.dumps(report, indent=2, default=str))
    if not verdict.passed:
        return report

    (out / "model").mkdir()
    torch.save(model.state_dict(), out / "model" / "weights.pt")
    (out / "model" / "config.yaml").write_text(json.dumps({"name": cfg["name"], "id_schema": cfg["id_schema"],
                                                            "model": cfg["model"]}, indent=2))
    (out / "calibration.json").write_text(json.dumps({"temperature": cal.temperature, "bias": cal.bias}))
    shutil.copytree(codebooks_dir, out / "codebooks")
    (out / "trie").mkdir()
    shutil.copy2(trie_path, out / "trie" / Path(trie_path).name)
    _, cb_meta = codebook_io.load(out / "codebooks")
    m = BundleManifest(
        bundle_id=bundle_id, backbone=backbone, adapter=None, id_schema=cfg["id_schema"], codebooks=cb_meta.version,
        codebook_sha256=cb_meta.sha256, trie_snapshot=f"trie_{trie_sha256[:12]}", trie_sha256=trie_sha256,
        corpus_snapshot=corpus_snapshot, corpus_epoch_range=epoch_range, reranker=os.environ.get("SIGIL_RERANKER", "rr_v1"),
        dataset=dataset, eval_report_uri=str(out / "eval_report.json"),
    )
    (m.signed(signing_key) if signing_key else m).save(out / "manifest.json")
    report["bundle_dir"] = str(out)
    return report


def main(argv=None) -> None:
    import torch
    from sigil_data.datasets import read
    from sigil_eval.golden_sets import load_qrels, load_queries
    from sigil_model.config import ModelConfig
    from sigil_model.encoder_decoder import SigilModel
    from sigil_model.tokenization import load_text_tokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--dataset", "--corpus", "--golden-queries", "--golden-qrels", "--trie", "--codebooks", "--bundle-id"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--config", default="configs/model/base.yaml")
    ap.add_argument("--incumbent")
    ap.add_argument("--out", default="artifacts/bundles")
    ap.add_argument("--steps", type=int, help="override every stage's step count (smoke runs)")
    a = ap.parse_args(argv)

    cfg = load_yaml(a.config)
    mcfg = ModelConfig.from_yaml(a.config)
    rows = list(read(a.dataset))
    ds_manifest = json.loads((Path(a.dataset) / "manifest.json").read_text())
    build_report = json.loads((Path(a.dataset) / "build_report.json").read_text())
    corpus = [json.loads(x) for x in Path(a.corpus).read_text(encoding="utf-8").splitlines() if x.strip()]
    queries, qrels = load_queries(a.golden_queries), load_qrels(a.golden_qrels)
    golden = [(queries[q], {d for d, g in rel.items() if g > 0}) for q, rel in qrels.items() if q in queries]
    trie_sha = Path(a.trie + ".sha256").read_text().strip()
    key = os.environ.get("SIGIL_BUNDLE_KEY")
    report = train_bundle(
        model=SigilModel(mcfg), tokenizer=load_text_tokenizer(mcfg.backbone_init), rows=rows, cfg=cfg, trie_path=a.trie,
        trie_sha256=trie_sha, codebooks_dir=a.codebooks, corpus=corpus, golden=golden,
        held_out=set(build_report["held_out_doc_ids"]), out_dir=a.out, bundle_id=a.bundle_id,
        backbone=mcfg.backbone_init, dataset=ds_manifest["version"], corpus_snapshot=ds_manifest["corpus_snapshots"][-1],
        epoch_range=(0, 2**62), incumbent=json.loads(Path(a.incumbent).read_text()) if a.incumbent else None,
        signing_key=key.encode() if key else None, steps=a.steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(report["offline_gates"]["summary"])
    raise SystemExit(0 if report["offline_gates"]["passed"] else 1)


if __name__ == "__main__":
    main()
