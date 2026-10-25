from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


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
        if self.vocab_size < 260:
            raise ValueError("vocab_size must be >= 260 for ByteTokenizer compatibility")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if min(self.n_layers, self.n_heads, self.n_kv_heads, self.hidden_dim, self.max_seq_len) < 1:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> ModelConfig:
        cfg = cls(**value)
        cfg.validate()
        return cfg


@dataclass
class TrainConfig:
    steps: int = 20
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

    def validate(self) -> None:
        if min(self.steps, self.batch_size, self.grad_accum, self.eval_every, self.save_every, self.eval_batches) < 1:
            raise ValueError("steps/batch/grad_accum/eval_every/save_every/eval_batches must be positive")
        if not 0.0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        if not 0.0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0,1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @classmethod
    def from_dict(cls, value: dict) -> TrainConfig:
        cfg = cls(**value)
        cfg.validate()
        return cfg


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return ModelConfig.from_dict(obj["model"]), TrainConfig.from_dict(obj["train"])
