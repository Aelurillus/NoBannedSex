"""Inspect a recipe without allocating model weights or provisioning compute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .model import NoBannedSexLM


def inspect_config(config, require_cuda=False):
    model_cfg, train_cfg = load_config(config)
    cuda = torch.cuda.is_available()
    if require_cuda and not cuda:
        raise RuntimeError("CUDA GPU unavailable; preflight never provisions an instance")
    if require_cuda and train_cfg.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("recipe requires BF16-capable CUDA hardware")
    with torch.device("meta"):
        model = NoBannedSexLM(model_cfg)
    parameters = model.parameter_count()
    tokens_per_step = train_cfg.batch_size * train_cfg.grad_accum * model_cfg.max_seq_len
    return {
        "config": str(config), "parameters": parameters,
        "tokens_per_step": tokens_per_step, "planned_training_tokens": tokens_per_step * train_cfg.steps,
        "fp32_adamw_parameter_gradient_moment_bytes": parameters * 16,
        "memory_estimate_exclusions": ["activations", "attention workspace", "CUDA context", "allocator", "checkpoint serialization"],
        "cuda_available": cuda, "gpu_name": torch.cuda.get_device_name(0) if require_cuda else None,
        "limitations": "Parameter-state estimate only; no VRAM-fit, throughput, convergence or cost guarantee. No compute provisioned.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--out")
    args = p.parse_args()
    report = inspect_config(args.config, args.require_cuda)
    content = json.dumps(report, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
