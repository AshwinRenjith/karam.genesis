# src/models/ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import GenesisConfig


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    output = (SiLU(gate(x)) ⊙ up(x)) → down
    Gradient never dies: SiLU derivative always positive.
    """
    def __init__(self, config: GenesisConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout   = nn.Dropout(p=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))   # [B, T, d_ff]
        up   = self.up_proj(x)             # [B, T, d_ff]
        hidden = self.dropout(gate * up)   # Hadamard product
        return self.down_proj(hidden)      # [B, T, d_model]
