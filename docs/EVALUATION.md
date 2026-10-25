# Experiment and evaluation contracts

Training loss does not establish factuality, dialogue quality, reasoning or
roleplay competence. Facts, uncertainty and fiction require different judgments.
The runtime does not apply content filters; the scoring code does not modify
model outputs or training data.

## Implemented measurements

The trainer records JSONL step metrics: cross-entropy, learning rate, gradient
norm, tokens seen, step time and training tokens/sec. Validation uses a separate
RNG and reports loss, perplexity (exponent capped at 80 to prevent overflow) and
next-token accuracy. CUDA adds peak allocated bytes, but this code path has not
been GPU-tested. `run_config.json` stores source commit/dirty flag, config, corpus
and tokenizer fingerprints, library/device details and parameter count.

```bash
python -m nobannedsex.evaluate --checkpoint runs/smoke/latest.pt \
  --suite data/eval-smoke.jsonl --out runs/smoke/evaluation.json \
  --lm-data data/eval-heldout.txt --max-new-tokens 16
```

Each JSONL short-answer example requires unique `id`, `domain`, `prompt`, boolean
`answerable`, and a nonempty list of reference `answers`. Prompt token counts must
fit the configured window. Oversized prompts fail; they are not silently cut to
produce misleading long-context results. Generation uses greedy decoding and
scores only newly generated tokens. Report includes per-example raw outputs,
checkpoint/suite hashes, decoding settings and per-domain metrics.

| Metric | Denominator / interpretation |
| --- | --- |
| Answerable exact match | Correct normalized short answers / answerable prompts |
| Answerable answer coverage | Nonempty non-abstention responses / answerable prompts |
| Answerable wrong-answer rate | Nonempty non-abstention exact-match failures / answerable prompts |
| Excessive abstention | Abstentions / answerable prompts |
| Unanswerable abstention | Abstentions / unanswerable prompts |

Normalization case-folds, collapses whitespace and removes terminal `.。!?！？`.
Abstention recognizes only exact normalized controlled strings: `unknown`,
`I don't know`, `insufficient information`, `不知道`, `信息不足`. This is a limited
short-answer scorer. It does not infer semantic correctness, confidence,
unsupported claims or arbitrary refusal language. Wrong-answer rate is not a
validated fabrication metric; verbose correct answers can fail exact match.
Empty responses have separate rates and do not count as answers or abstentions.
Missing denominator groups produce JSON `null`, not invented perfect scores.

The optional held-out LM pass scores every next-token target once, including the
partial final block, and weights loss by token count. It restarts context at each
block. The convenience text evaluator loads tokens into RAM; the Python scoring
function also accepts mmap token splits for larger validation corpora.

All bundled fixtures are synthetic tooling checks. They are too small to estimate
model quality. `eval-heldout.txt` is separate from `sample.txt`; no useful language
ability is expected from this smoke model. No external judge API is used.

## Experiment procedure

1. Freeze data rights/versions, dedup policy, tokenizer, exact model/training config,
   seed, source commit and test split before training. Keep eval data out of every
   training stage. Check overlap against prepared corpus audit hashes; near-duplicate
   contamination detection is still missing.
2. Run CPU tests and recovery acceptance. Record results and limitations.
3. On an already approved GPU, run the bounded synthetic benchmark, then a short
   real-data run. Compare end-to-end tokens/sec, VRAM, validation, finite gradients
   and recovery. Synthetic benchmarks exclude data transfer and are not cost forecasts.
4. Repeat meaningful experiments across seeds; compare fixed held-out suites and
   both accuracy and coverage. Preserve metrics, raw outputs and configs, not just
   a best score. Record provider invoices/GPU hours separately; no billing collector exists.

## Required future capability suites

- Factual claims: supported-claim precision, evidence grounding, time-sensitive
  fact handling, false-premise correction and answerable/unanswerable calibration.
- General tasks: held-out Chinese/English understanding, math, coding execution,
  extraction, translation, summarization and multi-turn conversation.
- Long context: evidence at multiple positions at 8K/32K/64K/128K, no-evidence and
  shuffled baselines, length-stratified accuracy. No such capability is demonstrated.
- Long text/fiction: human review of entities, timelines, repetition, plot and
  character consistency, style diversity and voluntary adult fictional settings.
  Invented events in fiction are not factual hallucinations. These are rubrics,
  not implemented automated quality scorers or a delivered roleplay model.
- Quantization: reference-versus-quantized task regressions, memory and speed.
  No working 4-bit/GGUF/GPTQ/AWQ path is included yet.
