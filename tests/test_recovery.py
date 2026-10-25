import json
from pathlib import Path

import pytest
import torch

from nobannedsex.checkpoint import file_sha256, load_checkpoint
from nobannedsex.train import train


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_resume_identical_to_uninterrupted(training_case, tmp_path):
    args, _ = training_case
    continuous = load_checkpoint(train(args))
    continuous_metrics = [json.loads(line) for line in (Path(args.out) / "metrics.jsonl").read_text().splitlines()]
    args.out = str(tmp_path / "interrupted")
    args.stop_after = 2
    cp_path = train(args)
    assert load_checkpoint(cp_path)["step"] == 2
    args.resume, args.stop_after = str(cp_path), None
    resumed = load_checkpoint(train(args))
    for key in ("model", "optimizer", "torch_rng_state", "python_rng_state", "train_rng_state", "eval_rng_state"):
        assert_nested_equal(continuous[key], resumed[key])
    metrics = [json.loads(line) for line in (Path(args.out) / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == [1, 2, 3, 4]
    for a, b in zip(continuous_metrics, metrics):
        for key in ("train_loss", "lr", "grad_norm", "val_loss", "val_next_token_accuracy"):
            assert a.get(key) == b.get(key)


@pytest.mark.parametrize("section,key,value", [("train", "batch_size", 1), ("train", "steps", 5),
                                               ("train", "seed", 3), ("model", "rope_theta", 5000)])
def test_resume_rejects_changed_recipe_before_writing(training_case, section, key, value):
    args, config = training_case
    args.stop_after = 2
    cp = train(args)
    digest = file_sha256(cp)
    config[section][key] = value
    Path(args.config).write_text(json.dumps(config))
    args.resume, args.stop_after = str(cp), None
    with pytest.raises(ValueError, match="config mismatch"):
        train(args)
    assert file_sha256(cp) == digest


def test_resume_rejects_changed_data(training_case, tmp_path):
    args, _ = training_case
    args.stop_after = 2
    cp = train(args)
    changed = tmp_path / "changed.txt"
    changed.write_text(Path(args.data).read_text() + "\n\nA new unseen document. " * 20)
    args.data, args.resume, args.stop_after = str(changed), str(cp), None
    with pytest.raises(ValueError, match="hash mismatch"):
        train(args)


def test_evaluation_does_not_change_training_randomness(training_case, tmp_path):
    args, config = training_case
    first = load_checkpoint(train(args))
    config["train"]["eval_every"] = 1
    Path(args.config).write_text(json.dumps(config))
    args.out = str(tmp_path / "more-eval")
    second = load_checkpoint(train(args))
    assert_nested_equal(first["model"], second["model"])
    assert_nested_equal(first["train_rng_state"], second["train_rng_state"])


def test_fresh_run_refuses_existing_output(training_case):
    args, _ = training_case
    train(args)
    with pytest.raises(ValueError, match="previous run"):
        train(args)


def test_stale_metrics_trimmed_to_checkpoint(training_case):
    args, _ = training_case
    args.stop_after = 2
    cp = train(args)
    metrics = Path(args.out) / "metrics.jsonl"
    with metrics.open("a") as f:
        f.write(json.dumps({"step": 3, "train_loss": 999}) + "\n")
    args.resume, args.stop_after = str(cp), None
    train(args)
    records = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [r["step"] for r in records] == [1, 2, 3, 4]
    assert records[2]["train_loss"] != 999


def test_gradient_accumulation_equivalent_to_full_batch(training_case, tmp_path):
    args, config = training_case
    config["model"]["dropout"] = 0
    config["train"].update(steps=1, warmup_steps=0, batch_size=2, grad_accum=2)
    Path(args.config).write_text(json.dumps(config))
    small = load_checkpoint(train(args))
    config["train"].update(batch_size=4, grad_accum=1)
    Path(args.config).write_text(json.dumps(config))
    args.out = str(tmp_path / "large-batch")
    large = load_checkpoint(train(args))
    for key in small["model"]:
        torch.testing.assert_close(small["model"][key], large["model"][key], atol=2e-7, rtol=1e-5)


def test_partial_final_log_record_does_not_block_recovery(training_case):
    args, _ = training_case
    args.stop_after = 2
    cp = train(args)
    metrics = Path(args.out) / "metrics.jsonl"
    with metrics.open("a") as f:
        f.write('{"step":3,"train_loss":')
    args.resume, args.stop_after = str(cp), None
    train(args)
    assert [json.loads(line)["step"] for line in metrics.read_text().splitlines()] == [1, 2, 3, 4]


def test_failed_checkpoint_write_preserves_previous(tmp_path, monkeypatch):
    from nobannedsex.checkpoint import atomic_save
    cp = tmp_path / "latest.pt"
    atomic_save({"model": {"x": torch.tensor([1.])}}, cp)
    digest = file_sha256(cp)
    def fail_save(*args, **kwargs):
        raise OSError("simulated interrupted checkpoint write")
    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated"):
        atomic_save({"new": 1}, cp)
    assert file_sha256(cp) == digest
    assert not list(tmp_path.glob("*.tmp"))
