import os
import torch
import torch.nn as nn
from src.config import GenesisConfig
from src.models.genesis import GenesisTransformer


def mitosis_slice_tensors(mother: GenesisTransformer, child: GenesisTransformer):
    """
    Performs Information-Preserving Weight Slicing.
    Extracts the weights of the 'child' model directly from the 'mother' model.
    Thanks to MRL (Matryoshka Representation Learning) and Head Dimension Invariance (64),
    this operation is mathematically sound.
    """
    mc = mother.config
    cc = child.config

    # Safety checks
    assert mc.head_dim == cc.head_dim == 64, "Head dimension must be exactly 64 for exact RoPE transfer."
    assert mc.vocab_size == cc.vocab_size, "Vocab size must be identical."
    assert cc.n_layers <= mc.n_layers, "Child cannot have more layers than Mother."
    assert cc.d_model <= mc.d_model, "Child d_model cannot exceed Mother d_model."
    assert cc.d_ff <= mc.d_ff, "Child d_ff cannot exceed Mother d_ff."
    assert cc.n_heads_q <= mc.n_heads_q, "Child n_heads_q cannot exceed Mother n_heads_q."
    assert cc.n_heads_kv == mc.n_heads_kv, "KV heads must remain constant for GQA ratio scaling."

    print(f"🧬 Initiating Mitosis: Slicing {mc.d_model}d Mother -> {cc.d_model}d Child")

    with torch.no_grad():
        # 1. Slice Token Embeddings (and implicitly LM Head, since they are weight-tied)
        child.tok_emb.weight.copy_(mother.tok_emb.weight[:, :cc.d_model])

        # 2. Slice Layers
        for l_idx in range(cc.n_layers):
            m_layer = mother.layers[l_idx]
            c_layer = child.layers[l_idx]

            # 2a. Attention QProjection
            # Shapes: [n_heads_q * 64, d_model]
            # We take the first 'cc.n_heads_q' heads, and slice the input dimension.
            c_layer.attn.wq.weight.copy_(
                m_layer.attn.wq.weight[:cc.n_heads_q * 64, :cc.d_model]
            )

            # 2b. Attention K, V Projections
            # KV heads are fully preserved (n_heads_kv is constant)
            # Shapes: [n_heads_kv * 64, d_model]
            c_layer.attn.wk.weight.copy_(m_layer.attn.wk.weight[:, :cc.d_model])
            c_layer.attn.wv.weight.copy_(m_layer.attn.wv.weight[:, :cc.d_model])

            # 2c. Attention Output Projection
            # Shape: [d_model, n_heads_q * 64]
            c_layer.attn.wo.weight.copy_(
                m_layer.attn.wo.weight[:cc.d_model, :cc.n_heads_q * 64]
            )

            # 2d. FFN Projections
            # Gate & Up: [d_ff, d_model]
            c_layer.ffn.gate_proj.weight.copy_(m_layer.ffn.gate_proj.weight[:cc.d_ff, :cc.d_model])
            c_layer.ffn.up_proj.weight.copy_(m_layer.ffn.up_proj.weight[:cc.d_ff, :cc.d_model])
            # Down: [d_model, d_ff]
            c_layer.ffn.down_proj.weight.copy_(m_layer.ffn.down_proj.weight[:cc.d_model, :cc.d_ff])

            # 2e. Layer Norms (Attention & FFN)
            # Shape: [d_model]
            c_layer.attn_norm.weight.copy_(m_layer.attn_norm.weight[:cc.d_model])
            c_layer.ffn_norm.weight.copy_(m_layer.ffn_norm.weight[:cc.d_model])

        # 3. Final RMSNorm
        child.norm.weight.copy_(mother.norm.weight[:cc.d_model])

        # 4. MTP Heads (if applicable, usually Child doesn't train MTP, but we copy anyway just in case)
        # We copy the RMSNorm, w1, w2 for the first MTP head if present.
        for h_idx in range(child.mtp_head.k):
            m_head = mother.mtp_head.heads[h_idx]
            c_head = child.mtp_head.heads[h_idx]
            c_head.norm.weight.copy_(m_head.norm.weight[:cc.d_model])
            c_head.w1.weight.copy_(m_head.w1.weight[:cc.d_model, :cc.d_model])
            c_head.w2.weight.copy_(m_head.w2.weight[:cc.d_model, :cc.d_model])

    return child


def spawn_expert(mother_checkpoint_path: str, domain_node_id: str) -> GenesisTransformer:
    """
    High-level API to spawn a new expert.
    Loads the Mother, initializes the Child layout, performs Mitosis, and returns the Child.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Init blank Mother
    m_config = GenesisConfig.mother()
    mother = GenesisTransformer(m_config)
    
    # Load Mother Weights
    if os.path.exists(mother_checkpoint_path):
        ckpt = torch.load(mother_checkpoint_path, map_location="cpu")
        mother.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    else:
        print(f"⚠️ Warning: mother checkpoint '{mother_checkpoint_path}' not found. Slicing from random initialization for testing.")
        
    # Init blank Child
    c_config = GenesisConfig.child(node_id=domain_node_id)
    child = GenesisTransformer(c_config)
    
    # Perform Slicing
    child = mitosis_slice_tensors(mother, child)
    
    print(f"✅ Spawned '{domain_node_id}' Child Node ({child.count_parameters()/1e6:.1f}M params)")
    return child.to(device)
