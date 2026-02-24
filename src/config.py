# src/config.py
from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class GenesisConfig:
    """
    Canonical configuration for a Genesis model node.
    ALL model dimensions flow from this config — never hardcode numbers.
    Head dim invariance:  head_dim = d_model // n_heads_q  (must equal 64)
    """
    # ── Architecture ──────────────────────────────────────────────────
    d_model: int = 768          # Mother: 768 | Child: 512
    n_layers: int = 16          # Mother: 16  | Child: 10
    n_heads_q: int = 12         # Mother: 12  | Child: 8
    n_heads_kv: int = 2         # Constant across Mother and all children
    d_ff: int = 4096            # Mother: 4096 | Child: 2688
    vocab_size: int = 32_000    # Shared tokenizer — NEVER CHANGE
    max_seq_len: int = 1024     # Shared — NEVER CHANGE
    rope_base: float = 10_000.0 # RoPE θ base — INVARIANT

    # ── Attention ─────────────────────────────────────────────────────
    swa_window: int = 256       # Sliding window for local layers
    n_local_layers: int = 12    # Layers 0–11: SWA attention
    # Layers 12–15: full causal attention (n_layers - n_local_layers)

    # ── Regularisation ────────────────────────────────────────────────
    dropout: float = 0.0        # 0 at pretrain, 0.05 at SFT
    norm_eps: float = 1e-6

    # ── MTP ───────────────────────────────────────────────────────────
    mtp_k: int = 4              # Predict k future tokens
    mtp_lambdas: list = field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.125]
    )

    # ── MRL ───────────────────────────────────────────────────────────
    mrl_dims: list = field(
        default_factory=lambda: [64, 128, 256, 384, 512, 768]
    )

    # ── Registry metadata ─────────────────────────────────────────────
    node_id: str = "mother"
    arch_version: str = "genesis-v1"
    is_child: bool = False
    parent_node_id: Optional[str] = None

    def __post_init__(self):
        assert self.d_model % self.n_heads_q == 0, "d_model must be divisible by n_heads_q"
        assert self.d_model // self.n_heads_q == 64, (
            f"head_dim must be exactly 64 for RoPE transfer. "
            f"Got {self.d_model // self.n_heads_q}. "
            f"Ensure d_model/n_heads_q = 64."
        )
        assert self.n_heads_q % self.n_heads_kv == 0, "n_heads_q must be divisible by n_heads_kv"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads_q  # Always 64

    @property
    def gqa_ratio(self) -> int:
        return self.n_heads_q // self.n_heads_kv

    @classmethod
    def mother(cls) -> "GenesisConfig":
        return cls()  # Default is Mother config

    @classmethod
    def child(cls, node_id: str, parent_id: str = "mother") -> "GenesisConfig":
        """Standard child config — 50M params."""
        return cls(
            d_model=512,
            n_layers=10,
            n_heads_q=8,
            n_heads_kv=2,
            d_ff=2688,
            node_id=node_id,
            is_child=True,
            parent_node_id=parent_id,
            n_local_layers=8,   # Local: 0–7, Global: 8–9
            dropout=0.1,
        )

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "GenesisConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)
