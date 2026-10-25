#!/usr/bin/env bash
# Runs ONLY on an existing user-provisioned GPU. Never creates cloud resources.
set -euo pipefail
if [[ "${GPU_TRAINING_APPROVED:-0}" != "1" ]]; then
  echo 'Explicit GPU training approval is required: GPU_TRAINING_APPROVED=1' >&2
  exit 1
fi
: "${BUDGET_CAP_USD:?Record the explicitly approved budget cap}"
: "${DATA:?Set DATA to the local authorized file or prepared dataset directory}"
: "${OUT:?Set OUT to a new persistent output directory, or the run being resumed}"
: "${CONFIG:=configs/gpu-benchmark.json}"
if [[ "${PERSISTENT_STORAGE_CONFIRMED:-0}" != "1" ]]; then
  echo 'Confirm durable storage and backup before training: PERSISTENT_STORAGE_CONFIRMED=1' >&2
  exit 1
fi
python - <<'PY'
import math
import os
value = float(os.environ['BUDGET_CAP_USD'])
if not math.isfinite(value) or value <= 0:
    raise SystemExit('BUDGET_CAP_USD must be finite and positive')
PY
python -m nobannedsex.preflight --config "$CONFIG" --require-cuda
if [[ ! -d "$(dirname "$OUT")" ]]; then
  echo 'Persistent output parent must already exist.' >&2
  exit 1
fi
args=(--config "$CONFIG" --data "$DATA" --out "$OUT" --device cuda)
if [[ -n "${TOKENIZER:-}" ]]; then args+=(--tokenizer "$TOKENIZER"); fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
# This label does not enforce provider billing. Configure a provider-side cap separately.
echo "Approved budget label: USD $BUDGET_CAP_USD; provider billing cap must be configured separately."
python -m nobannedsex.train "${args[@]}"
