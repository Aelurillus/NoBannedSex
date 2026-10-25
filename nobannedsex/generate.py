"""Minimal no-KV-cache autoregressive sampling, suitable only for small experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .model import NoBannedSexLM
from .tokenizer import load_tokenizer


def load_model(checkpoint: str, device: str):
    # For trusted local .pt files only. Never unpickle unknown third-party files.
    cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(cp["model_config"])
    model = NoBannedSexLM(cfg)
    model.load_state_dict(cp["model"])
    model.to(device).eval()
    return model


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
    path = Path(args.checkpoint)
    tok_path = args.tokenizer or (str(path.parent / "tokenizer.json") if (path.parent / "tokenizer.json").exists() else None)
    tk = load_tokenizer(tok_path)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    model = load_model(str(path), device)
    if tk.vocab_size != model.cfg.vocab_size:
        raise ValueError("Tokenizer/model vocab mismatch")
    ids = torch.tensor([tk.encode(args.prompt, bos=True)], device=device, dtype=torch.long)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k,
                         top_p=args.top_p, eos_id=tk.eos_id)
    print(tk.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
