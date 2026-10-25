# NoBannedSex v0.1 — Architecture and research roadmap

## What exists now

- Real, trainable decoder-only Transformer with random weight initialization.
- Causal scaled-dot-product attention with group-query heads (GQA), rotary positional encoding, RMSNorm, SwiGLU, and tied embedding/LM head weights.
- Two tokenizer modes: a lossless UTF-8 byte baseline for zero-dependency CPU tests; a self-trained ByteLevel BPE production tokenizer (`tokenizers` optional dependency).
- Single-process next-token training, document-level train/validation splitting, simple metric logging, optimizer checkpointing, exact corpus/tokenizer hash validation on resume, greedy/temperature/top-k/top-p sampling and safetensors export.
- No external hosted model, hidden safety filter, or moderation/keyword blacklist in the architecture.

## Known limitations

- The sample data and small smoke model are **engineering tests**, not useful conversational intelligence, factual alignment or roleplay models.
- `make_corpus` reads everything into RAM; it is unsuitable for large datasets.
- GQA currently materializes duplicated KV heads for compatibility and is not optimal for long contexts.
- Generation recomputes the context each token; no KV cache; no distributed training.
- GQA/RoPE by themselves do not make the model reliable at 128K context. Training from scratch at 128K is prohibitively expensive for early stages.
- No instruct fine-tuning, preference training, low-hallucination eval, roleplay tuning, 4-bit quantization kernels, or GGUF converter are implemented yet.
- Optimizer checkpoint (.pt) is a Python pickle file; only load trusted locally created files.

## Versioned plan

- v0.1: reproducible CPU smoke, repository-ready source, unit tests, cloud job skeleton (implemented; GitHub publication pending).
- v0.2: scalable data/tokenization, streaming token shards, benchmark baseline, single-GPU stability, FSDP.
- v0.3: meaningful pretrain experiments; ablations on hallucination/abstention vs task completion.
- v0.4: multi-domain instruction post-training, creative/roleplay and truthfulness datasets with distinct evaluation suites.
- v0.5: efficient long-context extension (8K -> 32K -> 128K only if compute/validation support it), retrieval tests, KV-cache optimized inference.
- v1.0: validated quantization and release with source, reproducibility manifests, tokenizer and model cards.

## Deployment concept

Codex / GitHub: engineering, tests, review, orchestration.
External cloud GPU instance(s): real training.
Durable object storage or attached persistent volume: corpora, checkpoints, experiment artifacts.
User-owned model files: independent inference and future training.

## Scaling caveat

`configs/research-280m.json` is a **shape candidate**, not a validated training recipe or promised performance/cost. It requires a custom BPE tokenizer of exactly 32K tokens and an authorized sufficiently large corpus. Add a distributed trainer before multiplying GPU count. Context length is currently set to 2048 to prevent accidental 128K quadratic-attention costs.
