# NoBannedSex — Codex instructions

This repository contains a **real, independently initialized language model**, not an API wrapper or pretrained-weights fine-tune. Do not introduce calls to OpenAI or any other hosted model in its training or inference pipeline.

## v0.1 objective

Make the project reproduce a correct CPU smoke test, then prepare larger runs to execute on **user-provisioned** cloud GPUs. Keep full training on external GPU infrastructure; do not assume Codex Cloud has an attached GPU or unlimited runtime.

## Guardrails for engineering quality

- Follow the user's open model design. Do not silently add output keyword filters or moderation classifiers.
- Keep truthfulness measurement separate from topic filtering. The model must distinguish uncertainty, speculation, fiction, and factual claims.
- Use legitimate, licensed/authorized training datasets; document provenance and deduplication. No copyrighted or private corpus should be silently downloaded.
- Never commit GPU API keys, SSH private keys, large datasets, checkpoints, or weights.
- Avoid treating training loss as evidence of model factuality, long-context understanding, or conversational quality.
- Any external cloud infrastructure spend requires an explicit budget cap from the user before launching.
- Do not claim long context, GGUF support, multimodal capability, censorship-free behavior, or low hallucination has been achieved before dedicated tests.

## First acceptance criteria

1. `pip install -e . pytest && pytest -q` succeeds on CPU.
2. `python -m nobannedsex.train --config configs/smoke.json --data data/sample.txt --out runs/smoke --device cpu` creates a checkpoint.
3. `python -m nobannedsex.generate --checkpoint runs/smoke/latest.pt --prompt '你好' --max-new-tokens 20` runs.
4. `python -m nobannedsex.export --checkpoint runs/smoke/latest.pt --out exports/smoke` creates a safetensors artifact when `safetensors` is installed.

## Prioritized follow-up tasks

1. Add proper document-sharded streaming dataset loader with per-document train/validation split, dedup and licensing manifests. Current loader is in-memory demo only.
2. Add fused/native GQA kernels, paged KV cache, and efficient inference. Current implementation duplicates KV heads and recomputes all context for every new token.
3. Build distributed GPU pretraining (`torchrun`/FSDP or equivalent) and recovery-safe optimizer sharding; current script is single-process.
4. Add independent truthfulness, multi-turn memory, roleplay, creative writing, coding and long-context evaluation datasets, with both answer and abstention metrics.
5. Add supervised post-training, preference optimization, long-context extension, and a working GPTQ/AWQ/GGUF conversion workflow after confirming tool compatibility.
6. Record measured VRAM, tokens/sec, GPU hours, total costs, and run reproducibility; report limitations openly.
