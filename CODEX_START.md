# Continue development

Read `AGENTS.md`, `docs/ARCHITECTURE.md`, `docs/DATA_PIPELINE.md`,
`docs/CLOUD_GPU.md`, `docs/EVALUATION.md` and `docs/VALIDATION.md` first.

The original v0.1 source has been imported; provenance is recorded in
`docs/SOURCE_PROVENANCE.md`. Do not replace the existing project with pretrained
weights or hosted-model API calls. Preserve the independent, no-hidden-topic-filter
design. Keep factuality and content policy separate.

CPU workflow:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m ruff check .
python -m pytest -q
bash scripts/cpu_acceptance.sh runs/new-acceptance
```

Use an empty output directory and a reviewable branch/PR. Do not commit weights,
corpora, secrets or temporary run directories. Update status documents when behavior
changes; include validation and limitations in the PR.

Next milestones: an explicitly approved bounded single-GPU experiment, near-duplicate
and benchmark contamination checks, stronger independent evaluation suites, distributed
training, efficient inference, long-context validation and quantization. Never launch
paid infrastructure without the user's explicit approval and spending cap.
