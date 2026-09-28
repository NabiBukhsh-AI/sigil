"""The training loop shared by every stage. §8.7.

ponytail: batches are counted in examples, not tokens. Queries are capped at 64 tokens,
so ``batch_tokens / 64`` examples is the conservative translation; switch to token
budgeting if document-side rows (512 tokens) start dominating a mix.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from sigil_core.ids import BASE, K
from sigil_training.objectives import DEFAULT_LEVEL_WEIGHTS, DEFAULT_MARGINS, proximal, rank_loss, seq_loss
from sigil_training.sam import SAM


@dataclass
class LoopConfig:
    steps: int
    lr: float = 1e-3
    lr_backbone: float = 1e-4
    warmup: int = 5000
    schedule: str = "inverse_sqrt"  # or "constant"
    batch_size: int = 256
    max_len: int = 64
    label_smoothing: float = 0.1
    level_weights: Sequence[float] = DEFAULT_LEVEL_WEIGHTS
    lambda_rank: float = 0.0
    margins: Sequence[float] = DEFAULT_MARGINS
    lambda_reg: float = 0.0
    sam: bool = False
    sam_rho: float = 0.05
    grad_clip: float = 1.0
    max_negatives: int = 8
    ckpt_every: int = 5000
    ckpt_dir: str | None = None
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    seed: int = 0
    log_every: int = 100
    history: list = field(default_factory=list)


def lr_factor(step: int, warmup: int, schedule: str) -> float:
    ramp = min(1.0, (step + 1) / max(warmup, 1))
    if schedule == "constant":
        return ramp
    return ramp * math.sqrt(max(warmup, 1) / max(step + 1, warmup, 1))


def collate(rows: Sequence[dict], tokenizer, max_len: int, max_negatives: int, device) -> dict:
    enc = tokenizer([r["input_text"] for r in rows], padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt")
    codes = torch.tensor([r["target_codes"] for r in rows], dtype=torch.long)
    L = codes.shape[1]
    negs = torch.zeros(len(rows), max_negatives, L, dtype=torch.long)
    mask = torch.zeros(len(rows), max_negatives, dtype=torch.bool)
    for i, r in enumerate(rows):
        hn = r.get("hard_negative_codes") or []
        for j, c in enumerate(hn[:max_negatives]):
            negs[i, j] = torch.tensor(c[:L])
            mask[i, j] = True
    batch = {
        "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "codes": codes,
        "tokens": codes + BASE + torch.arange(L) * K,
        "weights": torch.tensor([float(r.get("weight", 1.0)) for r in rows]),
        "neg": negs, "neg_mask": mask,
    }
    return {k: v.to(device) for k, v in batch.items()}


def train(
    model: torch.nn.Module,
    batches: Iterable[Sequence[dict]],
    tokenizer,
    cfg: LoopConfig,
    *,
    reference: list[torch.Tensor] | None = None,
    on_checkpoint: Callable[[int, torch.nn.Module], None] | None = None,
    device: str | None = None,
) -> list[dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    enc = [p for n, p in trainable if n.startswith("encoder.")]
    rest = [p for n, p in trainable if not n.startswith("encoder.")]
    groups = [g for g in ({"params": rest, "lr": cfg.lr}, {"params": enc, "lr": cfg.lr_backbone}) if g["params"]]
    opt = torch.optim.AdamW(groups, betas=cfg.betas, weight_decay=cfg.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    sam = SAM([p for _, p in trainable], opt, cfg.sam_rho) if cfg.sam else None
    params = [p for _, p in trainable]
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.autocast("cpu", enabled=False)

    def loss_of(b):
        with amp:
            logits = model(b["input_ids"], b["attention_mask"], b["tokens"])
        loss = seq_loss(logits, b["codes"], b["weights"], cfg.level_weights, cfg.label_smoothing)
        parts = {"seq": loss.item()}
        if cfg.lambda_rank and b["neg_mask"].any():
            r = rank_loss(logits, b["codes"], b["neg"], b["neg_mask"], cfg.margins)
            loss = loss + cfg.lambda_rank * r
            parts["rank"] = r.item()
        if cfg.lambda_reg and reference is not None:
            loss = loss + cfg.lambda_reg * proximal(params, reference)
        return loss, parts

    step = 0
    for rows in batches:
        if step >= cfg.steps:
            break
        for g, base in zip(opt.param_groups, base_lrs, strict=True):
            g["lr"] = base * lr_factor(step, cfg.warmup, cfg.schedule)
        b = collate(rows, tokenizer, cfg.max_len, cfg.max_negatives, device)
        loss, parts = loss_of(b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        if sam:
            sam.first_step()
            loss_of(b)[0].backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            sam.second_step()
        else:
            opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1
        if step % cfg.log_every == 0 or step == 1:
            cfg.history.append({"step": step, "loss": loss.item(), **parts})
        if cfg.ckpt_dir and step % cfg.ckpt_every == 0:
            Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(cfg.ckpt_dir) / f"step_{step}.pt")
            if on_checkpoint:
                on_checkpoint(step, model)
    model.eval()
    return cfg.history
