from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from nobannedsex.benchmark import benchmark
from nobannedsex.config import TrainConfig
from nobannedsex.preflight import inspect_config


ROOT = Path(__file__).resolve().parents[1]


def test_meta_preflight_research_shape_without_allocating_weights():
    report = inspect_config(ROOT / "configs/research-280m.json")
    assert 270_000_000 < report["parameters"] < 290_000_000
    assert report["tokens_per_step"] == 65536
    assert report["fp32_adamw_parameter_gradient_moment_bytes"] == 16 * report["parameters"]
    assert report["gpu_name"] is None


def test_cpu_benchmark_reports_real_timing(tmp_path):
    report = benchmark(ROOT / "configs/smoke.json", tmp_path / "benchmark.json", steps=2, warmup=1)
    assert report["measured_steps"] == 2
    assert report["tokens_per_second"] > 0
    assert report["peak_allocated_bytes"] is None
    assert json.loads((tmp_path / "benchmark.json").read_text())["kind"] == "synthetic-training-step-only"


def test_cloud_script_requires_explicit_approval():
    result = subprocess.run(["bash", "scripts/cloud_train.sh"], cwd=ROOT, capture_output=True, text=True,
                            env={"PATH": "/usr/bin:/bin", "GPU_TRAINING_APPROVED": "0"})
    assert result.returncode != 0
    assert "Explicit GPU training approval" in result.stderr


@pytest.mark.parametrize("change", [{"learning_rate": float("nan")}, {"warmup_steps": -1},
                                    {"weight_decay": -1}, {"grad_clip": 0}, {"grad_accum": 0},
                                    {"precision": "fp16"}, {"seed": -1}, {"num_threads": 0}])
def test_training_recipe_validation(change):
    cfg = replace(TrainConfig(), **change)
    with pytest.raises(ValueError):
        cfg.validate()
