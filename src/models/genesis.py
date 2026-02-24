# src/models/genesis.py
"""
GenesisTransformer: the unified Mother and Child model.
Configured entirely by GenesisConfig — no hardcoded dims anywhere.
"""
import torch
import torch.nn as nn
from src.config import GenesisConfig
from src.models.norm import RMSNorm
from src.models.attention import GenesisAttention, precompute_freqs_cis, \
    make_causal_mask, make_sliding_window_mask
from src.models.ffn import SwiGLUFFN
from src.models.mtp_head import MultiTokenPredictionHead
import math


class GenesisBlock(nn.Module):
    """One transformer layer: pre-norm GQA + pre-norm SwiGLU FFN."""
    def __init__(self, config: GenesisConfig, layer_idx: int):
        super().__init__()
        is_local = layer_idx < config.n_local_layers
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GenesisAttention(config, is_local=is_local)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                local_mask: torch.Tensor, global_mask: torch.Tensor) -> torch.Tensor:
        mask = local_mask if self.attn.is_local else global_mask
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GenesisTransformer(nn.Module):
    """
    KaramLLM Genesis Transformer.
    Works for both Mother (default config) and Child (child config).
    """
    def __init__(self, config: GenesisConfig):
        super().__init__()
        self.config = config

        # Token embedding + scale
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_scale = math.sqrt(config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [GenesisBlock(config, i) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # LM Head — weight-tied to embedding
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # Weight tying

        # MTP Head (used only during pretraining)
        self.mtp_head = MultiTokenPredictionHead(config, self.tok_emb.weight)

        # Precompute RoPE frequencies (persistent buffer)
        freqs_cis = precompute_freqs_cis(
            config.head_dim, config.max_seq_len, config.rope_base
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialise weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers) (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('down_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor,
                use_mtp: bool = False) -> dict:
        """
        token_ids: [B, T]
        use_mtp:   bool — True during pretraining, False at SFT/inference
        Returns: dict with 'logits' ([B, T, V]) and optionally 'mtp_logits' (list)
        """
        B, T = token_ids.shape
        device = token_ids.device

        # Build masks once per forward pass
        local_mask = make_sliding_window_mask(T, self.config.swa_window, device)
        global_mask = make_causal_mask(T, device)

        # Embed + scale
        x = self.tok_emb(token_ids) * self.emb_scale  # [B, T, d_model]

        # Forward through all layers
        for layer in self.layers:
            x = layer(x, self.freqs_cis, local_mask, global_mask)

        # Final norm
        h = self.norm(x)  # [B, T, d_model]

        # Standard LM logits (head 0)
        logits = self.lm_head(h)  # [B, T, vocab_size]

        result = {"logits": logits, "hidden": h}

        if use_mtp:
            result["mtp_logits"] = self.mtp_head(h)

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
