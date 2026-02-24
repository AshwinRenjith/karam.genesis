# src/models/attention.py
"""
Grouped Query Attention (GQA) with:
  • RoPE positional encoding (applied to Q and K only)
  • Sliding Window Attention for local layers (layers 0–11)
  • Full causal attention for global layers (layers 12–15)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import GenesisConfig


# ── RoPE Utilities ────────────────────────────────────────────────────────────

def precompute_freqs_cis(head_dim: int, max_seq_len: int, base: float = 10_000.0,
                          device: torch.device = None) -> torch.Tensor:
    """
    Precompute complex frequency tensor for RoPE.
    Returns: freqs_cis of shape [max_seq_len, head_dim // 2] (complex64)
    """
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # theta shape: [head_dim // 2]
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, theta)  # [seq_len, head_dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE rotation to Q and K.
    Args:
        xq: [B, T, n_heads_q, head_dim]
        xk: [B, T, n_heads_kv, head_dim]
        freqs_cis: [T, head_dim // 2] complex
    Returns:
        xq_rot, xk_rot — same shapes as inputs
    """
    # Reshape to complex: [B, T, n_heads, head_dim//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis broadcast: [1, T, 1, head_dim//2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    xq_rot = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_rot = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_rot.type_as(xq), xk_rot.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads from n_heads_kv to n_heads_q by repeating.
    x: [B, T, n_heads_kv, head_dim]
    → [B, T, n_heads_kv * n_rep, head_dim]
    """
    if n_rep == 1:
        return x
    B, T, n_kv, head_dim = x.shape
    x = x[:, :, :, None, :].expand(B, T, n_kv, n_rep, head_dim)
    return x.reshape(B, T, n_kv * n_rep, head_dim)


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular causal mask. True = KEEP, False = MASK."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def make_sliding_window_mask(seq_len: int, window: int,
                              device: torch.device) -> torch.Tensor:
    """
    Sliding window causal mask.
    Position i can attend to positions max(0, i-window) .. i inclusive.
    """
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    # Mask positions more than `window` steps back
    dist = torch.arange(seq_len, device=device).unsqueeze(0) - \
           torch.arange(seq_len, device=device).unsqueeze(1)
    # dist[i,j] = j - i  (negative = j is before i)
    window_mask = dist >= -window
    return causal & window_mask


# ── Grouped Query Attention ────────────────────────────────────────────────────

class GenesisAttention(nn.Module):
    """
    GQA with RoPE. Local vs global determined by `is_local` flag.
    """
    def __init__(self, config: GenesisConfig, is_local: bool):
        super().__init__()
        self.config = config
        self.is_local = is_local
        self.n_heads_q = config.n_heads_q
        self.n_heads_kv = config.n_heads_kv
        self.head_dim = config.head_dim  # Always 64
        self.n_rep = config.gqa_ratio
        self.window = config.swa_window

        # Projections — separate Q from KV for memory efficiency
        self.wq = nn.Linear(config.d_model, config.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_heads_kv * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_heads_kv * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads_q * self.head_dim, config.d_model, bias=False)

        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project
        xq = self.wq(x).view(B, T, self.n_heads_q, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_heads_kv, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_heads_kv, self.head_dim)

        # Apply RoPE to Q and K (V is position-free)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:T])

        # Expand KV to match Q heads
        xk = repeat_kv(xk, self.n_rep)  # [B, T, n_heads_q, head_dim]
        xv = repeat_kv(xv, self.n_rep)

        # Reshape for attention: [B, n_heads, T, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / scale  # [B, H, T, T]

        # Apply mask (False positions get -inf)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Attend to values
        out = torch.matmul(attn, xv)         # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, d_model]
        return self.wo(out)
