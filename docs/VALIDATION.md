# CPU validation — 2026-10-26 (Asia/Shanghai)

The repository initially contained README and Apache-2.0 LICENSE only (base
`de97cbb`). The original v0.1 archive was recovered and imported in `8bec9e9`.
Its unchanged six tests passed on CPU before the additional engineering work.
The existing full Apache-2.0 text is retained; no new license restrictions were added.

## Local environment and commands

- Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 2.3.5.
- pytest 9.1.1, tokenizers 0.23.3, safetensors 0.8.0, Ruff 0.17.2.
- CPU only; CUDA unavailable. No cloud resources created and no paid GPU launched.

```bash
python -m ruff check .
python -m compileall -q nobannedsex
bash -n scripts/*.sh
python -m pytest -q --junitxml=reports/junit.xml
bash scripts/cpu_acceptance.sh runs/acceptance-20261026
```

The test suite passes **65 tests**, including independent architecture equations,
causal masking, forward/backward, gradient-checkpoint equivalence with dropout,
accumulation, greedy/stochastic generation/EOS, byte/BPE tokenization, strict
checkpoint/recipe validation, CPU bitwise recovery for raw and mmap data,
partial-log recovery, atomic-save failure preservation, stable dedup/splits,
cross-shard windows, corruption checks, exports, CLI integration, scoring,
meta-device preflight, benchmark and GPU approval guard.

End-to-end CPU acceptance passed:

1. A 109,120-parameter random-initialized model trained on the synthetic fixture;
   stopped at step 4, then resumed with the same schedule through step 12.
2. Greedy generation ran and safetensors export completed. Tests separately reload
   the exported reference model and verify identical FP32 logits.
3. The local provenance manifest prepared checked token shards, then the same
   smoke config completed 12 steps using mmap data.
4. The evaluator wrote per-example/domain short-answer scores and a token-weighted
   held-out language-model report. The tiny model answered **0/3 answerable fixture
   questions correctly**. This is expected and is not evidence of useful ability.
5. A bounded synthetic CPU benchmark completed. Timing is environment-dependent;
   it is not a GPU throughput or cost forecast.

See [the machine-readable local summary](validation/cpu-20261026.json). Its original
acceptance measurements identify local import commit `d940cc1` with a dirty working
tree because they tested the subsequent changes before commit. No measurements
are attributed to a future commit or GPU. JUnit, run metadata, step metrics and
benchmark/evaluation JSON are also uploaded as CI artifacts; weights are excluded.

## Automated and unvalidated areas

The GitHub Actions workflow runs the CPU suite and full acceptance on Python
3.11/3.12 with PyTorch 2.6 CPU and pinned direct tooling dependencies. Remote job
status is available on the PR/Actions page; this document records local results
and does not pre-claim a successful remote run.

The 280M candidate was shape-checked on the meta device: **280,802,304 parameters**,
65,536 tokens/optimizer step. The partial FP32 AdamW state estimate is
4,492,836,864 bytes, excluding activations and other overhead. No actual 280M
weight allocation, GPU VRAM-fit test, cloud Docker build or CUDA run was performed.

Not demonstrated: practical dialogue/general-task competence, low hallucination,
long-context reasoning, long-form/roleplay quality, distributed training, efficient
KV-cached inference or 4-bit/GGUF/GPTQ/AWQ deployment. These remain research and
engineering milestones, not delivered model capabilities.
