# src/models/mtp_head.py
"""
Multi-Token Prediction (MTP) head.
During pretraining: predicts tokens t+1, t+2, t+3, t+4 simultaneously.
During inference:   only head[0] (k=1) is used — standard autoregressive.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.norm import RMSNorm
from src.config import GenesisConfig


class MTPHead(nn.Module):
    """One lightweight head predicting token at offset k."""
    def __init__(self, config: GenesisConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.w1 = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w2 = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, d_model]
        return self.w2(F.silu(self.w1(self.norm(h))))  # [B, T, d_model]


class MultiTokenPredictionHead(nn.Module):
    """
    K=4 prediction heads sharing the embedding matrix (weight-tied).
    Only head[0] is active at inference.
    """
    def __init__(self, config: GenesisConfig, embedding_weight: torch.Tensor):
        super().__init__()
        self.k = config.mtp_k
        self.lambdas = config.mtp_lambdas
        self.heads = nn.ModuleList([MTPHead(config) for _ in range(self.k)])
        # Store reference to embedding weight — NOT a parameter (weight-tied)
        self.register_buffer("embedding_weight", embedding_weight, persistent=False)

    def forward(self, h_final: torch.Tensor) -> list[torch.Tensor]:
        """
        h_final: [B, T, d_model] — output of final RMSNorm
        Returns: list of K logit tensors, shapes [B, T-k+1, vocab_size]
        """
        logit_list = []
        T = h_final.size(1)
        for k_idx, head in enumerate(self.heads):
            # Slice hidden states: predict for positions 0..T-k
            h_slice = h_final[:, :T - k_idx, :]  # [B, T-k, d_model]
            # MPS optimization: require memory contiguousness before heavy vocab mappings
            h_proj = head(h_slice).contiguous()   # [B, T-k, d_model]
            weight = self.embedding_weight.contiguous()
            logits = F.linear(h_proj, weight)     # [B, T-k, vocab]
            logit_list.append(logits)
        return logit_list
