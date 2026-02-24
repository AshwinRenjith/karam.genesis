import torch
import pytest
from src.config import GenesisConfig
from src.models.genesis import GenesisTransformer
from src.swarm.mitosis import mitosis_slice_tensors


def test_mitosis_slicing_shapes_and_invariance():
    """
    Test that slices don't cause dimension mismatches when passed through the Child model,
    and that head_dim strictly remains 64.
    """
    m_cfg = GenesisConfig.mother()
    mother = GenesisTransformer(m_cfg)
    
    c_cfg = GenesisConfig.child("test_child")
    child = GenesisTransformer(c_cfg)
    
    # Assert base invariances before slicing
    assert m_cfg.head_dim == 64
    assert c_cfg.head_dim == 64
    assert m_cfg.n_heads_kv == c_cfg.n_heads_kv == 2
    
    # Perform Mitosis
    child = mitosis_slice_tensors(mother, child)
    
    # 1. Verify parameter count matches intended sizing
    params = child.count_parameters()
    # 32K * 512 + ... = ~66.3M since we dynamically scaled d_ff up in Phase 1 to fix the PRD math
    assert 60_000_000 < params < 70_000_000, f"Child param count out of bounds: {params}"
    
    # 2. Verify forward pass computes without shape errors
    B, T = 2, 128
    ids = torch.randint(0, c_cfg.vocab_size, (B, T))
    
    # Should not throw any dimensional exceptions
    out = child(ids, use_mtp=False)
    
    assert out["logits"].shape == (B, T, c_cfg.vocab_size)
    assert out["hidden"].shape == (B, T, c_cfg.d_model)


def test_mitosis_weight_transfer():
    """
    Test that the exact values from the Mother's first N dimensions are fundamentally 
    transferred to the child.
    """
    m_cfg = GenesisConfig.mother()
    mother = GenesisTransformer(m_cfg)
    c_cfg = GenesisConfig.child("test_child")
    child = GenesisTransformer(c_cfg)
    
    child = mitosis_slice_tensors(mother, child)
    
    # Checking specific weights to ensure values match exactly up to the slice
    d = c_cfg.d_model
    v = c_cfg.vocab_size
    
    # Check embeddings
    assert torch.allclose(child.tok_emb.weight, mother.tok_emb.weight[:, :d])
    
    # Check Layer 0 Q-proj (first 8 heads)
    hq = c_cfg.n_heads_q * 64
    assert torch.allclose(child.layers[0].attn.wq.weight, mother.layers[0].attn.wq.weight[:hq, :d])
    
    # Check Layer 5 FFN Gate
    dff = c_cfg.d_ff
    assert torch.allclose(child.layers[5].ffn.gate_proj.weight, mother.layers[5].ffn.gate_proj.weight[:dff, :d])
    
