"""Nightly LoRA adapter refresh. §15.5, §35B R5.

    python pipelines/adapter_refresh.py --bundle artifacts/bundles/bundle_N --new-dataset datasets/ds_new \
        --replay-dataset datasets/ds_v14 --eval-new golden/new.jsonl --eval-old golden/old.jsonl \
        --adapter-version ad_v38 --trained-on cs_2026_09_07

1. Skip unless at least ``min_batch`` documents are waiting.
2. Train LoRA (rank 16, alpha 32) on decoder self-attention, cross-attention, and FFN only,
   with SAM and a proximal term, on new-document rows plus a 3:1 replay set stratified over
   level-1 codes.
3. Gate: old-document recall may regress by less than 1.0 point AND new-document recall
   must reach 0.6 of steady state. [FIXED] Rejection is the default outcome: the hot
   channel already covers new documents, so there is no reason to accept a damaging adapter.
4. On acceptance: write the adapter, move the documents to ACTIVE_LEARNED, and keep them in
   the hot lexical channel for 7 more days (a single refresh does not bring a new document
   to full parity).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sigil_decoding import DecodeConfig
from sigil_registry_client.records import DocState

from pipelines.full_train import evaluate_scorer

HOT_RETENTION = timedelta(days=7)


def refresh(*, model, tokenizer, trie, new_rows: list[dict], replay_pool: Sequence[dict],
            eval_new: list[tuple[str, set[str]]], eval_old: list[tuple[str, set[str]]], sid_to_doc: dict[str, str],
            lora_cfg: dict, lifecycle: dict, steps: int, out_dir: str, adapter_version: str, backbone: str,
            trained_on_snapshot: str, batch_size: int = 64, lambda_reg: float = 1e-3, device: str = "cpu") -> dict:
    import torch
    from sigil_data.datasets import batches
    from sigil_model.encoder_decoder import ModelScorer
    from sigil_model.lora import adapter_state_dict, apply_lora
    from sigil_training.loop import LoopConfig, train
    from sigil_training.replay import stratified_replay

    new_docs = {r["doc_id"] for r in new_rows}
    if len(new_docs) < lifecycle.get("adapter_min_batch", 2000):
        return {"decision": "skipped", "reason": f"{len(new_docs)} documents waiting, below min batch"}

    dcfg = DecodeConfig()
    scorer = ModelScorer(model, tokenizer, device)
    before_old = evaluate_scorer(scorer, trie, eval_old, sid_to_doc, dcfg)["recall@10"]
    before_new = evaluate_scorer(scorer, trie, eval_new, sid_to_doc, dcfg)["recall@10"]

    params = apply_lora(model, lora_cfg["rank"], lora_cfg["alpha"], lora_cfg["dropout"], lora_cfg["target_modules"])
    reference = [p.detach().clone() for p in params]
    rows = new_rows + stratified_replay(replay_pool, len(new_rows), lifecycle.get("replay_ratio", 3.0))
    cfg = LoopConfig(steps=steps, lr=1e-4, lr_backbone=0.0, warmup=min(500, steps // 10 + 1), schedule="constant",
                     batch_size=batch_size, sam=lora_cfg.get("sam_enabled", True), sam_rho=lora_cfg.get("sam_rho", 0.05),
                     lambda_reg=lambda_reg)
    train(model, batches(rows, batch_size), tokenizer, cfg, reference=reference, device=device)

    scorer = ModelScorer(model, tokenizer, device)
    after_old = evaluate_scorer(scorer, trie, eval_old, sid_to_doc, dcfg)["recall@10"]
    after_new = evaluate_scorer(scorer, trie, eval_new, sid_to_doc, dcfg)["recall@10"]
    regression = 100 * (before_old - after_old)
    steady = before_old  # learned documents' recall is the steady state new ones should approach
    accept = regression < 1.0 and after_new >= 0.6 * steady
    decision = {
        "decision": "accepted" if accept else "rejected", "old_doc_recall_regression_points": round(regression, 3),
        "cold_doc_recall_ratio": round(after_new / steady, 4) if steady else None,
        "new_doc_recall": {"before": before_new, "after": after_new}, "old_doc_recall": {"before": before_old, "after": after_old},
        "documents": sorted(new_docs),
    }
    if accept:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(adapter_state_dict(model), out / "adapter.pt")
        (out / "adapter.json").write_text(json.dumps({"version": adapter_version, "backbone": backbone,
                                                      "trained_on_snapshot": trained_on_snapshot, **decision}, indent=2))
    return decision


def mark_learned(registry, doc_uids: list[str], now: datetime | None = None) -> None:
    now = now or datetime.now(UTC)
    registry.set_state(doc_uids, DocState.ACTIVE_LEARNED)
    for u in doc_uids:
        registry.patch(u, metadata={"learned_at": now.isoformat()}, actor="adapter_refresh")


def sweep_hot_set(registry, hot_index, now: datetime | None = None, retention: timedelta = HOT_RETENTION) -> int:
    """Drop documents from the hot lexical channel once they have been learned for 7 days."""
    now = now or datetime.now(UTC)
    gone = [r for r in registry.records({DocState.ACTIVE_LEARNED})
            if "learned_at" in r.metadata and now - datetime.fromisoformat(r.metadata["learned_at"]) >= retention]
    hot_index.remove([str(r.semantic_id) for r in gone])
    return len(gone)


def main(argv=None) -> None:
    import torch
    from sigil_core.config import load_yaml
    from sigil_data.datasets import read
    from sigil_model.config import ModelConfig
    from sigil_model.encoder_decoder import SigilModel
    from sigil_model.tokenization import load_text_tokenizer
    from sigil_trie import TrieSnapshot

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--bundle", "--new-dataset", "--replay-dataset", "--eval-new", "--eval-old", "--adapter-version",
                 "--trained-on", "--corpus"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--config", default="configs/model/base.yaml")
    ap.add_argument("--serving", default="configs/serving/dev.yaml")
    ap.add_argument("--steps", type=int, default=20000)
    a = ap.parse_args(argv)

    cfg, serving = load_yaml(a.config), load_yaml(a.serving)
    bundle = Path(a.bundle)
    mcfg = ModelConfig.from_yaml(bundle / "model" / "config.yaml")
    model = SigilModel(mcfg)
    model.load_state_dict(torch.load(bundle / "model" / "weights.pt", map_location="cpu"))

    def pairs(path):
        rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        return [(r["query"], set(r["relevant"])) for r in rows]

    corpus = [json.loads(x) for x in Path(a.corpus).read_text(encoding="utf-8").splitlines() if x.strip()]
    trie_path = next((bundle / "trie").glob("*.trie"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with TrieSnapshot(trie_path) as trie:
        decision = refresh(
            model=model.to(device), tokenizer=load_text_tokenizer(mcfg.backbone_init), trie=trie,
            new_rows=list(read(a.new_dataset)), replay_pool=list(read(a.replay_dataset)), eval_new=pairs(a.eval_new),
            eval_old=pairs(a.eval_old), sid_to_doc={r["semantic_id"]: r["doc_uid"] for r in corpus}, lora_cfg=cfg["lora"],
            lifecycle=serving["lifecycle"], steps=a.steps, out_dir=str(bundle / "adapters" / a.adapter_version),
            adapter_version=a.adapter_version, backbone=mcfg.backbone_init, trained_on_snapshot=a.trained_on, device=device,
        )
    print(json.dumps({k: v for k, v in decision.items() if k != "documents"}, indent=2))
    raise SystemExit(0 if decision["decision"] != "rejected" else 1)


if __name__ == "__main__":
    main()
