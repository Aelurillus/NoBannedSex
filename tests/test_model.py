import pytest
import torch

from nobannedsex.config import ModelConfig
from nobannedsex.model import NoBannedSexLM


def tiny_cfg():
    return ModelConfig(vocab_size=260, dim=32, n_layers=2, n_heads=4,
                       n_kv_heads=2, hidden_dim=88, max_seq_len=16)


def test_model_forward_backward():
    torch.manual_seed(123)
    model = NoBannedSexLM(tiny_cfg())
    x = torch.randint(0, 260, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 260)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()


def test_causal_mask_no_future_leak():
    model = NoBannedSexLM(tiny_cfg()).eval()
    a = torch.randint(0, 260, (1, 8))
    b = a.clone()
    b[:, -1] = (b[:, -1] + 1) % 260
    with torch.no_grad():
        logits_a, _ = model(a)
        logits_b, _ = model(b)
    torch.testing.assert_close(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5, rtol=1e-5)


def test_generation_and_limits():
    model = NoBannedSexLM(tiny_cfg())
    out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=3, temperature=0)
    assert out.shape == (1, 6)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=260, dim=33, n_heads=4).validate()
