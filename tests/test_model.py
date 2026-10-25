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


def test_rmsnorm_reference_and_dtype():
    from nobannedsex.model import RMSNorm
    norm = RMSNorm(8)
    x = torch.randn(2, 3, 8)
    expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(norm(x), expected)
    assert norm(x.bfloat16()).dtype == torch.bfloat16


def test_rope_reference_norm_and_relative_position():
    from nobannedsex.model import apply_rope, rotary_cache
    x = torch.randn(1, 2, 7, 8)
    cos, sin = rotary_cache(7, 8, x.device, 10000.)
    result = apply_rope(x, cos, sin)
    angles = torch.arange(7).float()[:, None] / (10000 ** (torch.arange(4).float() / 4))
    a, b = x.chunk(2, -1)
    expected = torch.cat((a * angles.cos() - b * angles.sin(),
                          b * angles.cos() + a * angles.sin()), -1)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result.square().sum(-1), x.square().sum(-1))
    torch.testing.assert_close(result[:, :, 0], x[:, :, 0])
    # A shared positional shift leaves the Q/K inner product unchanged.
    q = torch.randn(1, 1, 1, 8).expand(1, 1, 7, 8)
    k = torch.randn(1, 1, 1, 8).expand(1, 1, 7, 8)
    qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    torch.testing.assert_close((qr[:, :, 1] * kr[:, :, 3]).sum(),
                               (qr[:, :, 3] * kr[:, :, 5]).sum())


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_gqa_matches_explicit_causal_attention(kv_heads):
    import math
    from nobannedsex.model import Attention, apply_rope, rotary_cache
    cfg = tiny_cfg()
    cfg.n_kv_heads = kv_heads
    attention = Attention(cfg).eval()
    x = torch.randn(2, 5, cfg.dim, requires_grad=True)
    batch, length, _ = x.shape
    head_dim = cfg.dim // cfg.n_heads
    q = attention.wq(x).view(batch, length, cfg.n_heads, head_dim).transpose(1, 2)
    k = attention.wk(x).view(batch, length, kv_heads, head_dim).transpose(1, 2)
    v = attention.wv(x).view(batch, length, kv_heads, head_dim).transpose(1, 2)
    cos, sin = rotary_cache(length, head_dim, x.device, cfg.rope_theta)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    heads = []
    for h in range(cfg.n_heads):
        group = h // (cfg.n_heads // kv_heads)
        score = q[:, h] @ k[:, group].transpose(-1, -2) / math.sqrt(head_dim)
        mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        probability = torch.softmax(score.masked_fill(mask, -torch.inf), -1)
        heads.append(probability @ v[:, group])
    expected = attention.wo(torch.stack(heads, 2).reshape(batch, length, cfg.dim))
    actual = attention(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    grad_a = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    grad_b = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(grad_a, grad_b, atol=1e-6, rtol=1e-5)


def test_swiglu_formula():
    from torch.nn import functional as F
    from nobannedsex.model import MLP
    mlp = MLP(tiny_cfg())
    x = torch.randn(2, 5, 32)
    gate = x @ mlp.gate.weight.T
    expected = (F.silu(gate) * (x @ mlp.up.weight.T)) @ mlp.down.weight.T
    torch.testing.assert_close(mlp(x), expected)


def test_gradient_checkpointing_equivalence():
    cfg = tiny_cfg()
    cfg.dropout = 0.2
    plain = NoBannedSexLM(cfg).train()
    cfg2 = type(cfg).from_dict({**cfg.to_dict(), "gradient_checkpointing": True})
    checkpointed = NoBannedSexLM(cfg2).train()
    checkpointed.load_state_dict(plain.state_dict())
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    torch.manual_seed(11)
    _, loss = plain(ids, ids)
    loss.backward()
    torch.manual_seed(11)
    _, loss2 = checkpointed(ids, ids)
    loss2.backward()
    torch.testing.assert_close(loss, loss2, atol=0, rtol=0)
    for left, right in zip(plain.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def test_tied_embedding_and_ignored_targets():
    from torch.nn import functional as F
    model = NoBannedSexLM(tiny_cfg())
    assert model.lm_head.weight is model.embed.weight
    ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, -100, 4]])
    logits, loss = model(ids, targets)
    expected = F.cross_entropy(logits[:, [0, 2]].reshape(-1, 260), torch.tensor([2, 4]))
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("change", [{"n_heads": 0}, {"n_kv_heads": 0}, {"dim": -8},
                                    {"rope_theta": 0}, {"rope_theta": float("nan")},
                                    {"dropout": 1.0}, {"n_layers": 1.5}])
def test_invalid_dimensions_raise_value_error(change):
    cfg = tiny_cfg()
    for name, value in change.items():
        setattr(cfg, name, value)
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize("kwargs", [{"max_new_tokens": -1}, {"temperature": float("nan")},
                                    {"top_k": -1}, {"top_p": 0}])
def test_sampling_invalid_arguments(kwargs):
    values = {"max_new_tokens": 2, **kwargs}
    with pytest.raises(ValueError):
        NoBannedSexLM(tiny_cfg()).generate(torch.tensor([[1]]), **values)


def test_sampling_eos_top_k_and_seed():
    model = NoBannedSexLM(tiny_cfg()).eval()
    ids = torch.tensor([[1, 2, 3]])
    greedy = model.generate(ids, 3, temperature=0)
    top_one = model.generate(ids, 3, temperature=0.9, top_k=1)
    assert torch.equal(greedy, top_one)
    torch.manual_seed(13)
    a = model.generate(ids, 4, top_k=20, top_p=0.8)
    torch.manual_seed(13)
    b = model.generate(ids, 4, top_k=20, top_p=0.8)
    assert torch.equal(a, b)
    first = int(greedy[0, 3])
    assert model.generate(ids, 9, temperature=0, eos_id=first).shape == (1, 4)
