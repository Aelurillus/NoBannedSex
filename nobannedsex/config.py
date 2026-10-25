from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


def positive_ints(obj, names):
    for name in names:
        value = getattr(obj, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


@dataclass
class ModelConfig:
    vocab_size: int = 260
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    hidden_dim: int = 176
    max_seq_len: int = 64
    rope_theta: float = 10000.0
    dropout: float = 0.0
    gradient_checkpointing: bool = False

    def validate(self) -> None:
        positive_ints(self, ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim", "max_seq_len"))
        if self.vocab_size < 260:
            raise ValueError("vocab_size must be >= 260 for the byte alphabet and special tokens")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not 0.0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> ModelConfig:
        cfg = cls(**value)
        cfg.validate()
        return cfg


@dataclass
class TrainConfig:
    steps: int = 20  # Immutable LR schedule horizon, including after resume.
    batch_size: int = 4
    grad_accum: int = 1
    learning_rate: float = 0.0005
    min_lr_ratio: float = 0.1
    warmup_steps: int = 2
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    eval_every: int = 5
    save_every: int = 5
    eval_batches: int = 2
    val_fraction: float = 0.15
    precision: str = "auto"
    num_threads: int = 4

    def validate(self) -> None:
        positive_ints(self, ("steps", "batch_size", "grad_accum", "eval_every", "save_every", "eval_batches", "num_threads"))
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be between 0 and steps")
        if not 0.0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        if not 0.0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0,1]")
        for name in ("learning_rate", "grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.precision not in ("auto", "fp32", "bf16"):
            raise ValueError("precision must be auto, fp32 or bf16")

    @classmethod
    def from_dict(cls, value: dict) -> TrainConfig:
        cfg = cls(**value)
        cfg.validate()
        return cfg


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(obj) != {"model", "train"}:
        raise ValueError("config must contain exactly model and train sections")
    return ModelConfig.from_dict(obj["model"]), TrainConfig.from_dict(obj["train"])
