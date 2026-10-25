from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def training_case(tmp_path):
    obj = {
        "model": {"vocab_size": 260, "dim": 32, "n_layers": 2, "n_heads": 4,
                  "n_kv_heads": 2, "hidden_dim": 48, "max_seq_len": 16, "dropout": 0.15},
        "train": {"steps": 4, "batch_size": 2, "grad_accum": 2, "warmup_steps": 1,
                  "eval_every": 3, "save_every": 2, "eval_batches": 2, "num_threads": 2},
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(obj))
    data = Path(__file__).resolve().parents[1] / "data" / "sample.txt"
    args = Namespace(config=str(config), data=str(data), tokenizer=None, out=str(tmp_path / "run"),
                     device="cpu", steps=None, stop_after=None, resume=None)
    return args, obj
