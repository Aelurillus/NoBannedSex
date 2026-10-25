"""Small, readable GQA/RoPE decoder-only Transformer, initialized from scratch."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight.to(x.dtype)


def rotary_cache(length: int, head_dim: int, device: torch.device, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    # RoPE uses paired halves (GPT-NeoX layout) consistently in Q and K.
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv)
    duplicated = torch.cat((angles, angles), dim=-1)
    return duplicated.cos()[None, None, :, :], duplicated.sin()[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.theta = cfg.rope_theta
        self.dropout = cfg.dropout
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.wq(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = rotary_cache(length, self.head_dim, x.device, self.theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # Explicit repetition is compatible across CPU and CUDA torch versions.
        # Production-scale kernels should replace this to avoid KV replication.
        repeats = self.n_heads // self.n_kv_heads
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class NoBannedSexLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied embeddings
        self.apply(self._init_weights)
        # Mild residual scaling stabilizes deep-from-scratch experiments.
        for block in self.layers:
            nn.init.normal_(block.attn.wo.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(block.ffn.down.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be rank-2 [batch, time]")
        if token_ids.size(0) == 0 or token_ids.size(1) == 0:
            raise ValueError("token_ids batch and time must be nonempty")
        if targets is not None and targets.shape != token_ids.shape:
            raise ValueError("targets shape must match token_ids")
        if token_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError(f"sequence exceeds configured max_seq_len={self.cfg.max_seq_len}")
        x = self.embed(token_ids)
        for block in self.layers:
            if self.cfg.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.inference_mode()
    def generate(self, ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 40, top_p: float = 1.0, eos_id: int | None = None) -> torch.Tensor:
        if ids.ndim != 2 or ids.size(0) != 1 or ids.size(1) == 0:
            raise ValueError("generation expects one nonempty [1, time] prompt")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        self.eval()
        for _ in range(max_new_tokens):
            context = ids[:, -self.cfg.max_seq_len:]
            logits, _ = self(context)
            logits = logits[:, -1, :].float()
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits = logits.masked_fill(logits < vals[:, [-1]], -float("inf"))
                if top_p < 1:
                    sorted_logits, sorted_indices = logits.sort(descending=True)
                    cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = (cumulative - F.softmax(sorted_logits, dim=-1)) >= top_p
                    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                    logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_indices, sorted_logits)
                next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat((ids, next_id), dim=1)
            if eos_id is not None and int(next_id.item()) == eos_id:
                break
        return ids
