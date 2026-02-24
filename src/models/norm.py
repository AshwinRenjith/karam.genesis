# src/models/norm.py
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    2× fewer ops than LayerNorm — no mean subtraction.
    Formula: x / RMS(x) * γ
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then back
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
