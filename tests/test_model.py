# tests/test_model.py
"""
Smoke tests — run these before launching any training.
All must pass. Any failure = architectural bug to fix first.
"""
import torch
import pytest
from src.config import GenesisConfig
from src.models.genesis import GenesisTransformer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_mother_shapes():
    cfg = GenesisConfig.mother()
    model = GenesisTransformer(cfg).to(get_device())
    ids = torch.randint(0, cfg.vocab_size, (2, 512)).to(get_device())
    out = model(ids, use_mtp=True)
    assert out["logits"].shape == (2, 512, cfg.vocab_size), f"Bad logit shape: {out['logits'].shape}"
    assert len(out["mtp_logits"]) == cfg.mtp_k
    assert out["mtp_logits"][0].shape == (2, 512, cfg.vocab_size)
    assert out["mtp_logits"][3].shape == (2, 509, cfg.vocab_size)  # T - 3
    print(f"✅ Mother shapes OK. Params: {model.count_parameters():,}")


def test_child_shapes():
    cfg = GenesisConfig.child("test_child")
    model = GenesisTransformer(cfg).to(get_device())
    ids = torch.randint(0, cfg.vocab_size, (2, 512)).to(get_device())
    out = model(ids, use_mtp=False)
    assert out["logits"].shape == (2, 512, cfg.vocab_size)
    print(f"✅ Child shapes OK. Params: {model.count_parameters():,}")


def test_head_dim_invariance():
    """Mother and child must share head_dim=64."""
    mother_cfg = GenesisConfig.mother()
    child_cfg = GenesisConfig.child("c1")
    assert mother_cfg.head_dim == 64
    assert child_cfg.head_dim == 64
    print("✅ head_dim invariance confirmed: 64 == 64")


def test_weight_tying():
    cfg = GenesisConfig.mother()
    model = GenesisTransformer(cfg)
    assert model.tok_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()
    print("✅ Embedding / LM head weight tying confirmed")


def test_parameter_count():
    cfg = GenesisConfig.mother()
    model = GenesisTransformer(cfg)
    n = model.count_parameters()
    # With d_ff=4096: ~202M total (within PRD sweet spot of ~277M)
    assert 195_000_000 < n < 210_000_000, f"Unexpected param count: {n:,}"
    print(f"✅ Parameter count: {n:,} (~{n/1e6:.1f}M)")


if __name__ == "__main__":
    test_mother_shapes()
    test_child_shapes()
    test_head_dim_invariance()
    test_weight_tying()
    test_parameter_count()
    print("\n🎉 All model tests passed.")
