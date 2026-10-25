"""Export model-only safetensors. Quantized inference kernels are a later milestone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .generate import load_bundle


def main():
    p = argparse.ArgumentParser(description="Export a trained model to safetensors")
    p.add_argument("--checkpoint", default="runs/smoke/latest.pt")
    p.add_argument("--out", default="exports/nobannedsex-v0.1")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    args = p.parse_args()
    try:
        from safetensors.torch import save_model
    except ImportError as exc:
        raise SystemExit('Install export dependency: pip install -e ".[export]"') from exc
    model, _ = load_bundle(args.checkpoint, "cpu")
    if args.dtype == "fp16":
        model.half()
    elif args.dtype == "bf16":
        model.bfloat16()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # save_model handles tied embedding/head tensors without duplicate storage.
    save_model(model, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2), encoding="utf-8")
    tok = Path(args.checkpoint).parent / "tokenizer.json"
    if tok.exists():
        shutil.copy2(tok, out / "tokenizer.json")
    else:
        (out / "tokenizer_type.txt").write_text("byte-v1; PAD=256,BOS=257,EOS=258,UNK=259\n")
    print(f"Exported {model.parameter_count():,} parameters to {out}")


if __name__ == "__main__":
    main()
