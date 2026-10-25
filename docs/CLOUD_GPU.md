# GPU handoff: prepared, not launched

No paid GPU instances have been started. These scripts never create, rent or
terminate cloud resources and have no provider API integration. CUDA training,
GPU recovery, throughput and VRAM fit have not been validated in this CPU task.

## Before approval

Run the CPU suite and acceptance script. Inspect a model shape without allocating
its parameters:

```bash
python -m nobannedsex.preflight --config configs/research-280m.json
```

The preflight counts parameters on PyTorch's meta device and reports a **partial**
FP32 AdamW state estimate: parameters, gradients and two optimizer moments (16
bytes/parameter). Activations, attention workspace, context, allocator and
serialization memory are excluded. This cannot determine whether a GPU fits.

`gpu-benchmark.json` is a bounded, small byte-tokenizer candidate (20 optimizer
steps). `research-280m.json` is a larger, unvalidated 2048-context shape candidate,
not an economical production recipe. It requires an exactly matching 32K BPE and
authorized corpus. Both are single-process; multiple visible GPUs do not create
distributed training. A GPU-enabled image and compatible NVIDIA host driver are
needed. The optional Dockerfile is an untested environment recipe, not a deployed
image. CPU CI deliberately installs a CPU PyTorch wheel.

## After explicit approval, on an existing GPU

Obtain an explicit budget cap and configure provider-side spending controls or a
short reservation. Arrange durable storage/backups and confirm training-data
rights before starting. The script's budget environment variable does **not**
enforce a provider billing cap. GPU rental may continue billing after Python exits;
the user must stop the rental through their provider.

Install a compatible CUDA PyTorch build, then `pip install -e '.[dev]'`. Clone the
reviewed commit and retain its exact environment. Run the preflight and bounded
synthetic benchmark on the existing GPU:

```bash
python -m nobannedsex.preflight --config configs/gpu-benchmark.json --require-cuda
python -m nobannedsex.benchmark --config configs/gpu-benchmark.json \
  --device cuda --steps 10 --warmup 2 --out /persistent/benchmark.json
```

No GPU measurements are promised by these commands. Only after inspecting real
results should you size a longer real-data experiment. A manual run requires all
of the following explicit inputs (substitute the actually approved cap):

```bash
export GPU_TRAINING_APPROVED=1
export BUDGET_CAP_USD='<explicitly-approved-cap>'
export PERSISTENT_STORAGE_CONFIRMED=1
export DATA=/persistent/authorized-corpus.txt
export OUT=/persistent/experiment-001
export CONFIG=configs/gpu-benchmark.json
bash scripts/cloud_train.sh
```

The small candidate requires enough distinct documents for a 512-token validation
partition. Research training uses a prepared corpus and your self-trained BPE:
set `DATA`, `TOKENIZER`, `CONFIG` accordingly. Prepare data with the exact split
seed/fraction in the chosen config (research candidate: seed 42, fraction 0.02).
No third-party data is downloaded automatically.

Recovery uses the same recipe and data with `RESUME=/persistent/experiment-001/latest.pt`.
Keep checkpoint, tokenizer and metrics together. Test a short interruption on
that GPU before a long run. The current bitwise guarantee is limited to tested
CPU recovery in the same environment; GPU and cross-device guarantees are absent.
Copy artifacts to another durable location before terminating the rental.
