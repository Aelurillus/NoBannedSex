# NoBannedSex

**Independent, open-source language model research. 从随机权重训练的独立语言模型研究项目。**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

> **Project status / 当前状态:** v0.1 research scaffold. This is **not** a trained, capable chat model or a released model checkpoint. The repository is being initialized; the training source code will be added separately. A locally tested prototype exists, but it has **not** been published to this repository yet.

## Vision / 项目目标

NoBannedSex aims to develop a **user-controlled, independently trained, quantizable general-purpose language model**. It is not an OpenAI API wrapper, prompt preset, or renamed third-party checkpoint.

The research priorities are:

- **Natural conversation and general tasks:** chat, writing, coding, reasoning, translation, document work and tool-use research.
- **Factual reliability:** distinguish known facts from guesses, uncertainty, memories, and fiction; minimize unsupported claims *without* making the model reflexively refuse to answer.
- **Creative freedom and roleplay:** flexible storytelling, character consistency, and optional lawful, consensual adult-oriented fictional scenarios.
- **Long-context competence:** progressively evaluate 8K/16K → 32K → 128K+ context, including effective retrieval and reasoning—not only a configurable context window.
- **Efficient local deployment:** investigate BF16, 8-bit and 4-bit inference, including conversion to widely supported formats where technically feasible.
- **Independent ownership:** training and inference must not rely on paid hosted foundation-model APIs.

**Openness principle:** the project does not plan a mandatory remote moderation service, keyword blacklist, or application-level topic filter inside the reference model runtime. This is a design goal, not a guarantee of any specific generation behavior. Factual calibration is a separate concern from topic filtering. Training-data rights, service-provider requirements, and applicable laws still matter.

## Current technical status / 当前技术状态

A **local v0.1 CPU research prototype** has passed six unit tests (6/6 on 2026-10-26). Its prototype implementation uses a randomly initialized decoder-only Transformer, causal attention, GQA, RoPE, RMSNorm, SwiGLU, byte-level or self-trained BPE tokenization, checkpoint resume, next-token pretraining, text sampling and safetensors export.

**Important:** the GitHub repository does not yet contain that prototype's source files. These statements describe the separately generated local prototype, not functionality verifiable by cloning the current repository. The tiny smoke checkpoint has no useful conversational, factual, or adult-roleplay ability.

| Research area | Status |
| --- | --- |
| Model architecture and CPU smoke tests | Local prototype tested; repository import pending |
| Production-scale tokenizer and dataset pipeline | Planned |
| Single-GPU training benchmark | Not run |
| Distributed GPU pretraining | Not implemented |
| Instruction tuning, creative and preference datasets | Not implemented |
| Truthfulness and abstention evaluation | Not implemented |
| Effective 128K context | Not demonstrated |
| 4-bit GGUF / GPTQ / AWQ inference release | Not implemented |
| Trained public model weights | None |

## Getting started (after source import)

The commands below apply **once the `nobannedsex/`, `configs/`, `data/`, `tests/` and `pyproject.toml` source files are committed**. They are not runnable against a README-only repository.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest -q
python -m nobannedsex.train --config configs/smoke.json --data data/sample.txt --out runs/smoke --device cpu
python -m nobannedsex.generate --checkpoint runs/smoke/latest.pt --prompt '你好' --max-new-tokens 30
python -m nobannedsex.export --checkpoint runs/smoke/latest.pt --out exports/smoke
```

**Smoke output will be meaningless** because it trains from scratch on a tiny test corpus. Never present this as model quality evidence.

## Compute plan / 算力方案

Development, unit testing, documentation and automation can run in Codex or an ordinary CPU environment. Actual pretraining needs rented cloud **GPU** instances (e.g. user-provisioned RunPod/Vast.ai), durable checkpoints, measurable throughput and an explicit spending limit. **No cloud GPU training has been launched by this project.**

## Roadmap / 路线图

1. **v0.1 — Engineering base:** publish the tested scaffold, CPU tests, checkpointing, documentation and CI.
2. **v0.2 — Data and GPU readiness:** legal/authorized corpus provenance, sharded streaming, robust tokenizer, single-GPU benchmarking.
3. **v0.3 — Measured pretraining:** compare small-from-scratch models against fixed quality, truthfulness and refusal/abstention metrics.
4. **v0.4 — Post-training:** multi-domain conversation, creative/roleplay, fact-grounding and distinct evaluation sets.
5. **v0.5 — Longer context and efficient inference:** long-context *evaluation*, caching, and model quantization.
6. **v1.0 — Validated release:** publish model artifacts only after measuring quality, performance and licensing compliance.

The roadmap is a plan, not a set of delivered capabilities.

## License / 许可证

**Project-owned source code and documentation** are licensed under the [Apache License, Version 2.0](LICENSE), allowing use, modification and redistribution, including commercial use, subject to its terms. The license includes patent-related provisions and does **not** remove requirements imposed by applicable law.

**Datasets, third-party components and future model weights are not automatically relicensed by this file.** Any weights and tokenizer releases will receive explicit license and provenance documentation when available. No pretrained weights or training dataset are published in the repository at this stage.

---

**Repository:** [Aelurillus/NoBannedSex](https://github.com/Aelurillus/NoBannedSex) · **Development status:** Early research, not a deployable conversational model.
