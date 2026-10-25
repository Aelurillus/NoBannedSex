"""Minimal no-KV-cache autoregressive sampling, suitable only for small experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import file_sha256, load_checkpoint
from .config import ModelConfig
from .model import NoBannedSexLM
from .tokenizer import load_tokenizer


def load_model(checkpoint: str, device: str):
    cp = load_checkpoint(checkpoint)
    cfg = ModelConfig.from_dict(cp["model_config"])
    model = NoBannedSexLM(cfg)
    model.load_state_dict(cp["model"])
    model.to(device).eval()
    return model


def load_bundle(checkpoint: str, device: str = "cpu", tokenizer_path: str | None = None):
    path = Path(checkpoint)
    if tokenizer_path is None and (path.parent / "tokenizer.json").exists():
        tokenizer_path = str(path.parent / "tokenizer.json")
    cp = load_checkpoint(path)
    digest = file_sha256(tokenizer_path) if tokenizer_path else None
    if digest != cp.get("tokenizer_sha256"):
        raise ValueError("tokenizer/checkpoint hash mismatch")
    tk = load_tokenizer(tokenizer_path)
    cfg = ModelConfig.from_dict(cp["model_config"])
    if cfg.vocab_size != tk.vocab_size:
        raise ValueError("tokenizer/model vocab mismatch")
    model = NoBannedSexLM(cfg)
    model.load_state_dict(cp["model"])
    return model.to(device).eval(), tk


def main():
    p = argparse.ArgumentParser(description="Sample text from an independently trained NoBannedSex checkpoint")
    p.add_argument("--checkpoint", default="runs/smoke/latest.pt")
    p.add_argument("--tokenizer", help="Defaults to tokenizer.json alongside checkpoint; otherwise byte tokenizer")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    model, tk = load_bundle(args.checkpoint, device, args.tokenizer)
    ids = torch.tensor([tk.encode(args.prompt, bos=True)], device=device, dtype=torch.long)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k,
                         top_p=args.top_p, eos_id=tk.eos_id)
    print(tk.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
