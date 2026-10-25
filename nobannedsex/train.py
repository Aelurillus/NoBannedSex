"""Single-process from-scratch pretraining with deterministic CPU recovery.

The schedule horizon is immutable. --stop-after simulates a clean interruption;
resume checks the complete recipe, data, tokenizer and runtime before writing.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time

import torch

from .checkpoint import FORMAT_VERSION, atomic_save, file_sha256, load_checkpoint
from .config import load_config
from .data import TokenSplit, iter_documents, open_dataset
from .model import NoBannedSexLM
from .tokenizer import load_tokenizer


def read_documents(path: str | Path) -> list[str]:
    return list(iter_documents(path))


def make_corpus(path, tokenizer, seq_len, val_fraction, seed):
    # Small-file convenience route only; use prepared shards for large corpora.
    docs = list(dict.fromkeys(read_documents(path)))
    if len(docs) < 2:
        raise ValueError("Need at least 2 unique documents for train/validation")
    random.Random(seed).shuffle(docs)
    count_val = max(1, min(len(docs) - 1, round(len(docs) * val_fraction)))
    val_docs, train_docs = docs[:count_val], docs[count_val:]

    def tokens_for(items):
        ids = [token for doc in items for token in tokenizer.encode(doc, eos=True)]
        if len(ids) < seq_len + 1:
            raise ValueError(f"split too short: need at least {seq_len + 1} tokens")
        return torch.tensor(ids, dtype=torch.long)

    return tokens_for(train_docs), tokens_for(val_docs), (len(train_docs), len(val_docs))


def sample_batch(tokens, batch_size, seq_len, device, generator=None):
    if len(tokens) < seq_len + 1:
        raise ValueError("split too short for sequence length")
    starts = torch.randint(0, len(tokens) - seq_len, (batch_size,), generator=generator)
    if isinstance(tokens, TokenSplit):
        windows = tokens.batch(starts.tolist(), seq_len + 1)
    else:
        windows = tokens[starts[:, None] + torch.arange(seq_len + 1)]
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


def lr_schedule(step, max_steps, warmup, peak, min_ratio):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    phase = min(1.0, max(0.0, (step - warmup) / max(1, max_steps - warmup)))
    return peak * (min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * phase)) / 2)


def evaluate(model, val_tokens, cfg, batch_size, device, autocast_ctx, generator=None):
    was_training = model.training
    model.eval()
    losses, correct, total = [], 0, 0
    try:
        with torch.no_grad():
            for _ in range(cfg.eval_batches):
                x, y = sample_batch(val_tokens, batch_size, model.cfg.max_seq_len, device, generator)
                with autocast_ctx():
                    logits, loss = model(x, y)
                losses.append(loss.item())
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.numel()
    finally:
        model.train(was_training)
    loss = sum(losses) / len(losses)
    return {"val_loss": loss, "val_perplexity": math.exp(min(loss, 80)),
            "val_next_token_accuracy": correct / total}


def save_checkpoint(path, model, optimizer, step, train_cfg, digest, tokenizer_digest,
                    train_generator, eval_generator, runtime):
    atomic_save({
        "format_version": FORMAT_VERSION, "step": step, "model_config": model.cfg.to_dict(),
        "train_config": asdict(train_cfg), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if runtime["device_type"] == "cuda" else None,
        "train_rng_state": train_generator.get_state(), "eval_rng_state": eval_generator.get_state(),
        "corpus_sha256": digest, "tokenizer_sha256": tokenizer_digest, "runtime": runtime,
    }, path)


def git_metadata():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def train(args):
    model_cfg, train_cfg = load_config(args.config)
    if args.steps is not None:
        train_cfg.steps = args.steps
        train_cfg.validate()
    stop_after = getattr(args, "stop_after", None)
    if stop_after is not None and not 1 <= stop_after <= train_cfg.steps:
        raise ValueError("stop_after must be between 1 and configured steps")
    target_step = stop_after or train_cfg.steps
    torch.set_num_threads(train_cfg.num_threads)
    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but CUDA is unavailable")
    bf16_available = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if train_cfg.precision == "bf16" and not bf16_available:
        raise ValueError("bf16 training requires a BF16-capable CUDA device")
    precision = "bf16" if train_cfg.precision != "fp32" and bf16_available else "fp32"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    runtime = {"device_type": device.type, "precision": precision, "torch_version": str(torch.__version__),
               "num_threads": train_cfg.num_threads}
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("output contains a previous run; use --resume or an empty directory")
    tokenizer_path = args.tokenizer
    if tokenizer_path is None and args.resume:
        candidate = Path(args.resume).parent / "tokenizer.json"
        tokenizer_path = str(candidate) if candidate.exists() else None
    tk = load_tokenizer(tokenizer_path)
    if model_cfg.vocab_size != tk.vocab_size:
        raise ValueError("model/tokenizer vocab mismatch")
    tokenizer_digest = file_sha256(tokenizer_path) if tokenizer_path else None
    if Path(args.data).is_dir():
        train_tokens, val_tokens, manifest, digest = open_dataset(args.data, tk, model_cfg.max_seq_len)
        if manifest["tokenizer_sha256"] != tokenizer_digest:
            raise ValueError("dataset/tokenizer hash mismatch")
        if manifest["seed"] != train_cfg.seed or manifest["val_fraction"] != train_cfg.val_fraction:
            raise ValueError("dataset split seed/fraction must match train config")
        counts = (manifest["documents"]["train"], manifest["documents"]["validation"])
    else:
        digest = file_sha256(args.data)
        train_tokens, val_tokens, counts = make_corpus(args.data, tk, model_cfg.max_seq_len,
                                                      train_cfg.val_fraction, train_cfg.seed)
    model = NoBannedSexLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate,
                                  weight_decay=train_cfg.weight_decay, betas=(0.9, 0.95))
    train_generator = torch.Generator().manual_seed(train_cfg.seed + 1)
    eval_generator = torch.Generator().manual_seed(train_cfg.seed + 2)
    start_step = 0
    if args.resume:
        cp = load_checkpoint(args.resume)
        if cp.get("format_version") != FORMAT_VERSION:
            raise ValueError("legacy v1 checkpoints support inference only; restart to use validated recovery")
        if cp["model_config"] != model_cfg.to_dict():
            raise ValueError("resume model config mismatch")
        if cp["train_config"] != asdict(train_cfg):
            raise ValueError("resume train config mismatch; schedule horizon and sampling recipe are immutable")
        if cp["corpus_sha256"] != digest or cp["tokenizer_sha256"] != tokenizer_digest:
            raise ValueError("resume corpus/tokenizer hash mismatch")
        if cp["runtime"] != runtime:
            raise ValueError("resume runtime mismatch; use the same device type, precision and PyTorch version")
        start_step = cp["step"]
        if target_step <= start_step:
            raise ValueError("resume target must exceed checkpoint step")
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        torch.set_rng_state(cp["torch_rng_state"])
        random.setstate(cp["python_rng_state"])
        train_generator.set_state(cp["train_rng_state"])
        eval_generator.set_state(cp["eval_rng_state"])
        if device.type == "cuda":
            if cp["cuda_rng_states"] is None or len(cp["cuda_rng_states"]) != torch.cuda.device_count():
                raise ValueError("resume CUDA device count mismatch")
            torch.cuda.set_rng_state_all(cp["cuda_rng_states"])
    # Validate everything before copying tokenizer or touching an existing run.
    out.mkdir(parents=True, exist_ok=True)
    if tokenizer_path and Path(tokenizer_path).resolve() != (out / "tokenizer.json").resolve():
        shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    if not tokenizer_path and (out / "tokenizer.json").exists():
        raise ValueError("unexpected tokenizer in output directory")
    metrics_path = out / "metrics.jsonl"
    if args.resume and metrics_path.exists():
        # Discard log records written after the last durable checkpoint.
        records = []
        lines = metrics_path.read_text().splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not line.endswith("\n"):
                    break  # A crash can leave a partially written final log record.
                raise
            if record["step"] <= start_step:
                records.append(record)
        metrics_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    config_used = {"model": model_cfg.to_dict(), "train": asdict(train_cfg),
                   "corpus_sha256": digest, "tokenizer_sha256": tokenizer_digest,
                   "train_docs": counts[0], "val_docs": counts[1], "runtime": runtime,
                   "python": platform.python_version(), "git": git_metadata(),
                   "parameters": model.parameter_count(), "start_step": start_step,
                   "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    (out / "run_config.json").write_text(json.dumps(config_used, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "params": model.parameter_count(),
                      "train_tokens": len(train_tokens), "validation_tokens": len(val_tokens),
                      "start_step": start_step, "target_step": target_step}), flush=True)

    def ac():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()

    model.train()
    for step in range(start_step, target_step):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        lr = lr_schedule(step, train_cfg.steps, train_cfg.warmup_steps,
                         train_cfg.learning_rate, train_cfg.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(train_cfg.grad_accum):
            x, y = sample_batch(train_tokens, train_cfg.batch_size, model_cfg.max_seq_len, device, train_generator)
            with ac():
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss; last durable checkpoint preserved")
            (loss / train_cfg.grad_accum).backward()
            losses.append(loss.detach().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        tokens = train_cfg.batch_size * train_cfg.grad_accum * model_cfg.max_seq_len
        record = {"step": step + 1, "train_loss": sum(losses) / len(losses),
                  "lr": lr, "grad_norm": float(grad_norm), "step_seconds": seconds,
                  "train_tokens_per_second": tokens / seconds, "tokens_seen": (step + 1) * tokens}
        # A clean interruption must not introduce an extra validation RNG draw.
        if (step + 1) % train_cfg.eval_every == 0 or step + 1 == train_cfg.steps:
            record.update(evaluate(model, val_tokens, train_cfg, train_cfg.batch_size, device, ac, eval_generator))
        if device.type == "cuda":
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
        if (step + 1) % train_cfg.save_every == 0 or step + 1 == target_step:
            save_checkpoint(out / "latest.pt", model, optimizer, step + 1, train_cfg, digest, tokenizer_digest,
                            train_generator, eval_generator, runtime)
    return out / "latest.pt"


def main():
    p = argparse.ArgumentParser(description="Train NoBannedSex from random initial weights")
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--data", default="data/sample.txt", help="text/JSONL file or prepared dataset directory")
    p.add_argument("--tokenizer", help="self-trained tokenizer.json; omit for byte baseline")
    p.add_argument("--out", default="runs/smoke")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--steps", type=int, help="override schedule horizon for a NEW experiment")
    p.add_argument("--stop-after", type=int, help="stop at this completed step without changing schedule horizon")
    p.add_argument("--resume", help="path to a project-owned checkpoint")
    train(p.parse_args())


if __name__ == "__main__":
    main()
