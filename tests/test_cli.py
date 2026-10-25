import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from safetensors.torch import load_model as load_safetensors

from nobannedsex.checkpoint import file_sha256, load_checkpoint
from nobannedsex.config import ModelConfig
from nobannedsex.generate import load_bundle, load_model
from nobannedsex.model import NoBannedSexLM
from nobannedsex.tokenizer import BPETokenizer, train_bpe
from nobannedsex.train import train


ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run([sys.executable, "-m", *args], check=True, capture_output=True, text=True,
                          cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})


def test_cli_train_resume_generate_export_and_eval(tmp_path):
    out = tmp_path / "run"
    config = json.loads((ROOT / "configs/smoke.json").read_text())
    config["train"].update(steps=3, warmup_steps=0, num_threads=2)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config))
    common = ["--config", str(cfg_path), "--data", "data/sample.txt", "--out", str(out), "--device", "cpu"]
    cli("nobannedsex.train", *common, "--stop-after", "1")
    cp = out / "latest.pt"
    cli("nobannedsex.train", *common, "--resume", str(cp))
    assert load_checkpoint(cp)["step"] == 3
    generated = cli("nobannedsex.generate", "--checkpoint", str(cp), "--prompt", "你好", "--max-new-tokens", "3",
                    "--temperature", "0", "--device", "cpu")
    assert "你好" in generated.stdout
    export = tmp_path / "export"
    cli("nobannedsex.export", "--checkpoint", str(cp), "--out", str(export))
    assert (export / "model.safetensors").exists()
    cfg = ModelConfig.from_dict(json.loads((export / "config.json").read_text()))
    recovered = NoBannedSexLM(cfg).eval()
    load_safetensors(recovered, str(export / "model.safetensors"))
    original = load_model(str(cp), "cpu")
    ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(original(ids)[0], recovered(ids)[0], atol=0, rtol=0)
    report = tmp_path / "eval.json"
    cli("nobannedsex.evaluate", "--checkpoint", str(cp), "--suite", "data/eval-smoke.jsonl", "--out", str(report),
        "--max-new-tokens", "3")
    assert json.loads(report.read_text())["metrics"]["count"] == 4


def test_self_trained_bpe_training_and_tokenizer_integrity(training_case, tmp_path):
    import pytest
    args, config = training_case
    tokenizer = tmp_path / "tokenizer.json"
    train_bpe([args.data], str(tokenizer), vocab_size=300)
    tk = BPETokenizer(tokenizer)
    text = "中文 with café 🎉"
    assert tk.decode(tk.encode(text, bos=True, eos=True)) == text
    config["model"]["vocab_size"] = tk.vocab_size
    Path(args.config).write_text(json.dumps(config))
    args.tokenizer, args.stop_after = str(tokenizer), 1
    cp = train(args)
    _, saved = load_bundle(str(cp), "cpu")
    assert saved.vocab_size == tk.vocab_size
    altered = tmp_path / "same-vocab-different-hash.json"
    altered.write_bytes(tokenizer.read_bytes() + b"\n")
    assert file_sha256(altered) != file_sha256(tokenizer)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(str(cp), "cpu", str(altered))
    cli("nobannedsex.export", "--checkpoint", str(cp), "--out", str(tmp_path / "bpe-export"), "--dtype", "bf16")
    assert (tmp_path / "bpe-export/tokenizer.json").read_bytes() == tokenizer.read_bytes()
