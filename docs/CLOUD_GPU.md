# Cloud GPU handoff — no personal GPU required

1. Create your own account with a GPU provider (e.g., RunPod or Vast.ai). Cloud GPU compute is distinct from Codex Cloud, and provider billing is separate.
2. Rent **one** suitable CUDA GPU for first experiments. Check price, VRAM, CPU/RAM/disk, host driver compatibility and regional availability yourself. Do not buy a multi-GPU instance for an untested prototype.
3. Mount a persistent volume or object store. Never put the only copy of a checkpoint on ephemeral scratch disk.
4. Clone the git repository or upload this project package to the GPU instance and execute:

```bash
pip install -e '.[tokenizer,export]'
python -m nobannedsex.tokenizer --files /workspace/persistent/corpus.txt --out /workspace/persistent/tokenizer.json --vocab-size 32000
DATA=/workspace/persistent/corpus.txt \
TOKENIZER=/workspace/persistent/tokenizer.json \
OUT=/workspace/persistent/nobannedsex-run \
CONFIG=configs/research-280m.json \
bash scripts/cloud_train.sh
```

5. `configs/research-280m.json` is **not** a production-ready economical training config. Benchmark VRAM, tokens/sec, data quality, loss curves and checkpoint recovery with a tiny portion first. Adjust parameters and use the right corpus size before long experiments.
6. Resume carefully with `python -m nobannedsex.train ... --resume /workspace/persistent/nobannedsex-run/latest.pt`. The data and tokenizer must be byte-for-byte identical to the original job; the program verifies hashes.
7. Set a spending cap/alerts, save logs, back up checkpoints, and terminate the GPU rental after data is safely saved.

**Important:** Always install a CUDA-enabled PyTorch wheel compatible with the provider's GPU and driver. CPU-only PyTorch will not use GPU resources. This scaffold is single-GPU, single-process; scaling to multiple GPUs requires additional distributed-training implementation.
