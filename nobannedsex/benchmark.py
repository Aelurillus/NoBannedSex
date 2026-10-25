"""Bounded synthetic training-step benchmark on an already available CPU/GPU."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import time

import torch

from .checkpoint import file_sha256
from .config import load_config
from .model import NoBannedSexLM
from .train import git_metadata


def benchmark(config, out, device="cpu", steps=5, warmup=2):
    if not 1 <= steps <= 1000 or not 0 <= warmup <= 100:
        raise ValueError("benchmark steps must be 1..1000 and warmup 0..100")
    model_cfg, cfg = load_config(config)
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; benchmark does not provision compute")
    supports_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if cfg.precision == "bf16" and not supports_bf16:
        raise ValueError("recipe requires a BF16-capable CUDA device")
    bf16 = cfg.precision != "fp32" and supports_bf16
    model = NoBannedSexLM(model_cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                                  betas=(.9, .95))
    times = []
    # Data transfer is deliberately excluded; use real-data trainer metrics for end-to-end throughput.
    x = torch.randint(model_cfg.vocab_size, (cfg.batch_size, model_cfg.max_seq_len + 1), device=device)
    for step in range(warmup + steps):
        if device.type == "cuda":
            if step == warmup:
                torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            ac = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
            with ac:
                _, loss = model(x[:, :-1], x[:, 1:])
            (loss / cfg.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if step >= warmup:
            times.append(elapsed)
    tokens_per_step = cfg.batch_size * cfg.grad_accum * model_cfg.max_seq_len
    report = {"kind": "synthetic-training-step-only", "config_sha256": file_sha256(config),
              "model_config": model_cfg.to_dict(), "train_config": cfg.__dict__,
              "device": str(device), "precision": "bf16" if bf16 else "fp32",
              "torch_version": str(torch.__version__), "python": platform.python_version(), "git": git_metadata(),
              "parameters": model.parameter_count(), "measured_steps": steps, "warmup_steps": warmup,
              "seconds_per_step": sum(times) / len(times),
              "tokens_per_second": tokens_per_step * steps / sum(times),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "limitations": "Synthetic data, no model quality measurement, no provisioning, no cloud cost estimate."}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()
    print(json.dumps(benchmark(args.config, args.out, args.device, args.steps, args.warmup), indent=2))


if __name__ == "__main__":
    main()
