from argparse import Namespace
import json
from pathlib import Path

import torch

from nobannedsex.tokenizer import ByteTokenizer
from nobannedsex.train import train
from nobannedsex.generate import load_model


def test_utf8_roundtrip():
    tk = ByteTokenizer()
    txt = "你好, world! 🎈"
    assert tk.decode(tk.encode(txt, bos=True, eos=True)) == txt


def test_training_resume(tmp_path):
    sample = Path(__file__).parent.parent / "data" / "sample.txt"
    cfg = json.loads((Path(__file__).parent.parent / "configs" / "smoke.json").read_text())
    cfg["model"].update({"dim": 32, "n_layers": 1, "n_heads": 4,
                         "n_kv_heads": 2, "hidden_dim": 88, "max_seq_len": 16})
    cfg["train"].update({"steps": 1, "batch_size": 2, "eval_every": 1,
                         "save_every": 1, "eval_batches": 1, "warmup_steps": 0})
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    root = tmp_path / "run"
    args = Namespace(config=str(cfg_path), data=str(sample), tokenizer=None,
                     out=str(root), device="cpu", steps=None, resume=None)
    cp = train(args)
    assert cp.exists()
    model = load_model(str(cp), "cpu")
    assert model.parameter_count() > 0
    assert torch.isfinite(model(torch.tensor([[2, 3]]))[0]).all()
    args.steps = 2
    args.resume = str(cp)
    train(args)
    checkpoint = torch.load(cp, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 2
