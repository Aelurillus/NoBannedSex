"""Single-process pretraining MVP. This is NOT a distributed or production training stack."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

import torch

from .config import load_config
from .model import NoBannedSexLM
from .tokenizer import load_tokenizer


def read_documents(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix == ".jsonl":
        docs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                    raise ValueError("JSONL lines must each contain a string 'text' field")
                if item["text"].strip():
                    docs.append(item["text"])
        return docs
    return [s for s in path.read_text(encoding="utf-8").split("\n\n") if s.strip()]


def make_corpus(path: str | Path, tokenizer, seq_len: int, val_fraction: float, seed: int):
    docs = read_documents(path)
    if len(docs) < 2:
        raise ValueError("Need at least 2 documents (blank-line-separated paragraphs or JSONL records) for train/validation")
    rnd = random.Random(seed)
    rnd.shuffle(docs)
    count_val = max(1, min(len(docs) - 1, round(len(docs) * val_fraction)))
    val_docs, train_docs = docs[:count_val], docs[count_val:]

    def tokens_for(items):
        ids = []
        for doc in items:
            ids.extend(tokenizer.encode(doc, eos=True))
        if len(ids) < seq_len + 2:
            raise ValueError(f"Split has only {len(ids)} tokens; need {seq_len+2}. Provide more data or reduce context")
        return torch.tensor(ids, dtype=torch.long)

    return tokens_for(train_docs), tokens_for(val_docs), (len(train_docs), len(val_docs))


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    starts = torch.randint(0, len(tokens) - seq_len, (batch_size,))
    windows = tokens[starts[:, None] + torch.arange(seq_len + 1)]
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


def lr_schedule(step: int, max_steps: int, warmup: int, peak: float, min_ratio: float):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    phase = min(1.0, max(0.0, (step - warmup) / max(1, max_steps - warmup)))
    return peak * (min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * phase)) / 2)


def evaluate(model, val_tokens, cfg, batch_size, device, autocast_ctx):
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = sample_batch(val_tokens, batch_size, model.cfg.max_seq_len, device)
            with autocast_ctx():
                logits, loss = model(x, y)
            losses.append(loss.item())
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
    model.train()
    return {"val_loss": sum(losses) / len(losses), "val_next_token_accuracy": correct / total}


def save_checkpoint(path: Path, model, optimizer, step: int, model_config: dict,
                    train_config: dict, corpus_sha256: str, tokenizer_sha256: str | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step, "model_config": model_config, "train_config": train_config,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "corpus_sha256": corpus_sha256, "tokenizer_sha256": tokenizer_sha256,
    }
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def train(args):
    model_cfg, train_cfg = load_config(args.config)
    if args.steps is not None:
        train_cfg.steps = args.steps
        train_cfg.validate()
    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer
    if tokenizer_path:
        # Preserve the actual training tokenizer with the checkpoints.
        destination = out / "tokenizer.json"
        if Path(tokenizer_path).resolve() != destination.resolve():
            shutil.copyfile(tokenizer_path, destination)
        tokenizer_path = str(destination)
    elif (out / "tokenizer.json").exists():
        tokenizer_path = str(out / "tokenizer.json")

    tk = load_tokenizer(tokenizer_path)
    if model_cfg.vocab_size != tk.vocab_size:
        raise ValueError(f"Config vocab_size {model_cfg.vocab_size} != tokenizer vocab_size {tk.vocab_size}")
    digest = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    tokenizer_digest = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest() if tokenizer_path else None
    train_tokens, val_tokens, counts = make_corpus(args.data, tk, model_cfg.max_seq_len,
                                                   train_cfg.val_fraction, train_cfg.seed)

    model = NoBannedSexLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate,
                                  weight_decay=train_cfg.weight_decay, betas=(0.9, 0.95))
    start_step = 0
    if args.resume:
        # WARNING: torch.load with weights_only=False is for YOUR OWN trusted checkpoints only.
        cp = torch.load(args.resume, map_location="cpu", weights_only=False)
        if cp["model_config"] != model_cfg.to_dict():
            raise ValueError("Resume model config mismatch")
        if cp["corpus_sha256"] != digest or cp["tokenizer_sha256"] != tokenizer_digest:
            raise ValueError("Resume corpus/tokenizer hash mismatch")
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        start_step = cp["step"]
        torch.set_rng_state(cp["torch_rng_state"])
        if device.type == "cuda" and cp.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(cp["cuda_rng_states"])

    config_used = {"model": model_cfg.to_dict(), "train": asdict(train_cfg),
                   "corpus_sha256": digest, "tokenizer_sha256": tokenizer_digest,
                   "train_docs": counts[0], "val_docs": counts[1]}
    (out / "run_config.json").write_text(json.dumps(config_used, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "params": model.parameter_count(),
                      "train_tokens": len(train_tokens), "validation_tokens": len(val_tokens),
                      "start_step": start_step, "target_steps": train_cfg.steps}))

    def ac():
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    model.train()
    for step in range(start_step, train_cfg.steps):
        lr = lr_schedule(step, train_cfg.steps, train_cfg.warmup_steps,
                         train_cfg.learning_rate, train_cfg.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(train_cfg.grad_accum):
            x, y = sample_batch(train_tokens, train_cfg.batch_size, model_cfg.max_seq_len, device)
            with ac():
                _, loss = model(x, y)
            (loss / train_cfg.grad_accum).backward()
            losses.append(loss.detach().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        record = {"step": step + 1, "train_loss": sum(losses) / len(losses),
                  "lr": lr, "grad_norm": float(grad_norm)}
        if (step + 1) % train_cfg.eval_every == 0 or step + 1 == train_cfg.steps:
            record.update(evaluate(model, val_tokens, train_cfg, train_cfg.batch_size, device, ac))
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if (step + 1) % train_cfg.save_every == 0 or step + 1 == train_cfg.steps:
            save_checkpoint(out / "latest.pt", model, optimizer, step + 1, model_cfg.to_dict(),
                            asdict(train_cfg), digest, tokenizer_digest)
    return out / "latest.pt"


def main():
    p = argparse.ArgumentParser(description="Train NoBannedSex from random initial weights")
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--data", default="data/sample.txt")
    p.add_argument("--tokenizer", help="Optional tokenizer.json from tokenizer training. Omit for byte tokenizer")
    p.add_argument("--out", default="runs/smoke")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--steps", type=int, help="Override training steps")
    p.add_argument("--resume", help="Path to OWN trusted training checkpoint")
    train(p.parse_args())


if __name__ == "__main__":
    main()
