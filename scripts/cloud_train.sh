#!/usr/bin/env bash
# Usage: DATA=/workspace/persistent/corpus.txt OUT=/workspace/persistent/run bash scripts/cloud_train.sh
set -euo pipefail
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('ERROR: no CUDA GPU available. Use a GPU instance and matching PyTorch CUDA build.')
print('GPU:', torch.cuda.get_device_name(0))
print('BF16:', torch.cuda.is_bf16_supported())
PY
: "${DATA:?Set DATA=/path/to/authorized-training-corpus.txt}"
: "${OUT:?Set OUT=/path/to/persistent/output-directory}"
: "${CONFIG:=configs/research-280m.json}"
: "${TOKENIZER:?Set TOKENIZER=/path/to/trained/tokenizer.json}"
if [[ ! -d "$(dirname "$OUT")" ]]; then
  echo "ERROR: output parent directory must exist and be persistent: $(dirname "$OUT")" >&2
  exit 1
fi
python -m nobannedsex.train --config "$CONFIG" --data "$DATA" --tokenizer "$TOKENIZER" --out "$OUT" --device cuda
