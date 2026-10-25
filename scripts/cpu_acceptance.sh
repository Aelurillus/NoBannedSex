#!/usr/bin/env bash
set -euo pipefail
# Use an empty output directory; never silently overwrite prior experiments.
TASK_RUN_DIR="${1:-runs/cpu-acceptance}"
python -m nobannedsex.preflight --config configs/research-280m.json
python -m nobannedsex.train --config configs/smoke.json --data data/sample.txt --out "$TASK_RUN_DIR/raw" --device cpu --stop-after 4
python -m nobannedsex.train --config configs/smoke.json --data data/sample.txt --out "$TASK_RUN_DIR/raw" --device cpu --resume "$TASK_RUN_DIR/raw/latest.pt"
python -m nobannedsex.generate --checkpoint "$TASK_RUN_DIR/raw/latest.pt" --prompt '你好' --max-new-tokens 20 --temperature 0 --device cpu
python -m nobannedsex.export --checkpoint "$TASK_RUN_DIR/raw/latest.pt" --out "$TASK_RUN_DIR/export"
python -m nobannedsex.data --sources data/sources.sample.json --out "$TASK_RUN_DIR/dataset" --val-fraction 0.15 --shard-tokens 257
python -m nobannedsex.train --config configs/smoke.json --data "$TASK_RUN_DIR/dataset" --out "$TASK_RUN_DIR/sharded" --device cpu
python -m nobannedsex.evaluate --checkpoint "$TASK_RUN_DIR/raw/latest.pt" --suite data/eval-smoke.jsonl --out "$TASK_RUN_DIR/raw/evaluation.json" --lm-data data/eval-heldout.txt --max-new-tokens 8
python -m nobannedsex.benchmark --config configs/smoke.json --device cpu --steps 3 --warmup 1 --out "$TASK_RUN_DIR/raw/benchmark.json"
