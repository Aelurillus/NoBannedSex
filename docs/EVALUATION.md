# Measurement plan: avoid conflating truthfulness with refusal

Track held-out evaluations by *task domain*, *language*, *context length*, *answerability*, *ground truth* and *date*. Never train on the test set.

Proposed suites:

1. Factual QA: supported-claim precision, fabrication rate and correctly answered coverage.
2. Unanswerable prompts: appropriate uncertainty rate, incorrect confident answers, and excessive refusal on answerable questions.
3. Contradiction / false premises: contradiction detection and qualified correction.
4. Long context: answer accuracy at 8K, 32K, 64K, 128K by evidence location; compare against shuffled/no-evidence baselines.
5. Long-form generation: consistency of entities/timelines, repetition, and section coverage.
6. General ability: Chinese/English language understanding, math, coding tests, extraction and summarization.
7. Adult conversational and fictional roleplay: voluntary adult settings only; separate ratings for character consistency, naturalness, plot coherence and stylistic diversity. Avoid conflating creative invention with factual hallucination.
8. Robustness: reproducibility of outputs, data contamination checks, model regression tests after quantization.

Prefer reporting both factual accuracy and answer rate. A model that only answers 'I don't know' is not automatically truthful or useful.

This v0.1 repository measures only next-token validation loss and next-token accuracy (language modeling indicators). It has *not* demonstrated low hallucination or any state-of-the-art capability.
