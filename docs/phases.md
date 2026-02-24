# KaramLLM Genesis — Build Phases
## From Zero to Production: The Complete Code-Level Implementation Guide

> **Source Architecture:** `KARAM_GENESIS_FINAL.md`  
> **Target:** KaramLLM v3 "Genesis" — 202M parameters, M1 MacBook Air  
> **Language:** Python 3.11 + PyTorch 2.3 (MPS backend)  
> **Status:** IMPLEMENTATION READY

---

## Project Directory Layout

Every file referenced in these phases maps to this exact tree. Create this scaffold before Phase 0.

```
karam.v1/
├── phases.md                      ← this file
├── KARAM_GENESIS_FINAL.md         ← architecture PRD
├── requirements.txt
├── setup.py
│
├── src/
│   ├── __init__.py
│   ├── config.py                  ← GenesisConfig dataclass
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── genesis.py             ← GenesisTransformer (Mother / Child)
│   │   ├── attention.py           ← GQA + RoPE + SWA
│   │   ├── ffn.py                 ← SwiGLU FFN
│   │   ├── norm.py                ← RMSNorm
│   │   └── mtp_head.py            ← MultiTokenPredictionHead
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretrain.py            ← Phase 1: MRL + MTP pretraining loop
│   │   ├── sft.py                 ← Phase 2: Supervised fine-tuning
│   │   ├── dpo.py                 ← Phase 3: DPO alignment
│   │   └── losses.py              ← mrl_loss(), mtp_loss(), distill_loss(), dpo_loss()
│   │
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   └── train_tokenizer.py     ← Phase 0: SentencePiece BPE
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pretrain_loader.py     ← FineWeb-Edu streaming dataloader
│   │   ├── sft_loader.py          ← ChatML conversation formatter
│   │   └── dpo_loader.py          ← Chosen/rejected pair loader
│   │
│   ├── swarm/
│   │   ├── __init__.py
│   │   ├── mitosis.py             ← Phase 5: weight slicing + distillation
│   │   ├── router.py              ← Learned SwarmRouter
│   │   ├── registry.py            ← NodeRegistry (JSON-backed)
│   │   └── lifecycle.py           ← Apoptosis + SLERP Fusion
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── generate.py            ← Autoregressive + speculative decoding
│   │   ├── kv_cache.py            ← Persistent KV cache
│   │   └── quantize.py            ← INT4/INT8 quantization wrapper
│   │
│   └── utils/
│       ├── __init__.py
│       ├── checkpoint.py          ← save/load with versioning
│       ├── device.py              ← M1 MPS detection + fallback
│       └── memory_monitor.py      ← live RAM tracker
│
├── scripts/
│   ├── train_tokenizer.sh
│   ├── run_pretrain.sh
│   ├── run_sft.sh
│   ├── run_dpo.sh
│   └── run_mitosis.sh
│
├── tests/
│   ├── test_model.py
│   ├── test_losses.py
│   ├── test_mitosis.py
│   └── test_router.py
│
├── api/
│   ├── main.py                    ← FastAPI server
│   └── schemas.py
│
└── checkpoints/                   ← auto-created during training
    ├── tokenizer/
    ├── pretrain/
    ├── sft/
    ├── dpo/
    └── children/
```

---

## Phase 0 — Project Bootstrap & Tokenizer (Day 1–2)

### 0.1 `requirements.txt`

```
torch>=2.3.0
sentencepiece>=0.2.0
datasets>=2.19.0
transformers>=4.41.0
sentence-transformers>=3.0.0
tqdm>=4.66.0
wandb>=0.17.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0
numpy>=1.26.0
```

### 0.2 `src/config.py` — The Single Source of Truth

```python
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
    d_ff: int = 2048            # Mother: 2048 | Child: 1344
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
            d_ff=1344,
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
```

### 0.3 `src/tokenizer/train_tokenizer.py`

```python
# src/tokenizer/train_tokenizer.py
"""
Train a 32K SentencePiece BPE tokenizer on FineWeb-Edu.
Streams 1M sentences — never writes full dataset to disk.
Runtime: ~2 hours on M1.
"""
import io
import sentencepiece as spm
from datasets import load_dataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

VOCAB_SIZE = 32_000
OUTPUT_DIR = Path("checkpoints/tokenizer")
MODEL_PREFIX = str(OUTPUT_DIR / "karam_spm_32k")
N_SENTENCES = 1_000_000


def stream_fineweb_sentences(n: int) -> io.StringIO:
    """Stream n sentences from FineWeb-Edu into an in-memory buffer."""
    log.info(f"Streaming {n:,} sentences from FineWeb-Edu...")
    buf = io.StringIO()
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    count = 0
    for sample in dataset:
        # Each sample is a document — take first 3 sentences per doc
        text = sample["text"]
        sentences = text.split(". ")[:3]
        for s in sentences:
            s = s.strip()
            if len(s) > 20:  # Filter noise
                buf.write(s + "\n")
                count += 1
                if count >= n:
                    break
        if count >= n:
            break
    log.info(f"Streamed {count:,} sentences.")
    buf.seek(0)
    return buf


def train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write sentences to a temp file (spm requires a file path)
    tmp_path = OUTPUT_DIR / "train_corpus.txt"
    buf = stream_fineweb_sentences(N_SENTENCES)
    tmp_path.write_text(buf.read())
    log.info(f"Corpus written to {tmp_path}")

    # Train the tokenizer
    spm.SentencePieceTrainer.train(
        input=str(tmp_path),
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9999,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<mask>"],  # id=4
        num_threads=8,
        input_sentence_size=N_SENTENCES,
        shuffle_input_sentence=True,
    )
    log.info(f"Tokenizer saved: {MODEL_PREFIX}.model")

    # Verify
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{MODEL_PREFIX}.model")
    test = "The quick brown fox jumps over the lazy dog."
    tokens = sp.Encode(test, out_type=str)
    ids = sp.Encode(test)
    log.info(f"Test encode: {tokens}")
    log.info(f"Test IDs: {ids}")
    log.info(f"Vocab size confirmed: {sp.GetPieceSize()}")
    assert sp.GetPieceSize() == VOCAB_SIZE

    # Cleanup temp corpus
    tmp_path.unlink()
    log.info("✅ Tokenizer training complete.")


if __name__ == "__main__":
    train()
```

**Validation:** After running, `checkpoints/tokenizer/karam_spm_32k.model` must exist and `sp.GetPieceSize() == 32000`.

---

## Phase 1 — Model Architecture (Day 1–3)

Build the complete `GenesisTransformer` bottom-up: primitives first, then blocks, then the full model. Every function has a shape assertion so bugs surface at construction time, not after 12 hours of training.

### 1.1 `src/models/norm.py`

```python
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
```

### 1.2 `src/models/attention.py`

```python
# src/models/attention.py
"""
Grouped Query Attention (GQA) with:
  • RoPE positional encoding (applied to Q and K only)
  • Sliding Window Attention for Mother (202M, 16 layers, d=768):
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
```

### 1.3 `src/models/ffn.py`

```python
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
```

### 1.4 `src/models/mtp_head.py`

```python
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
            h_proj = head(h_slice)                # [B, T-k, d_model]
            logits = h_proj @ self.embedding_weight.T  # [B, T-k, vocab]
            logit_list.append(logits)
        return logit_list
```

### 1.5 `src/models/genesis.py` — The Core Model

```python
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
```

### 1.6 `tests/test_model.py` — Run This Before Training

```python
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
    # Should be ~202M — alert if >215M or <195M
    assert 200_000_000 < n < 220_000_000, f"Unexpected param count: {n:,}"
    print(f"✅ Parameter count: {n:,} (~{n/1e6:.1f}M)")


if __name__ == "__main__":
    test_mother_shapes()
    test_child_shapes()
    test_head_dim_invariance()
    test_weight_tying()
    test_parameter_count()
    print("\n🎉 All model tests passed.")
```

---

## Phase 2 — Loss Functions (Day 3)

All training losses in one file. Each function is independently testable.

### 2.1 `src/training/losses.py`

```python
# src/training/losses.py
"""
All training loss functions for the Genesis pipeline.

  mrl_loss()      — Matryoshka Representation Learning
  mtp_loss()      — Multi-Token Prediction
  pretrain_loss() — MRL + MTP combined
  sft_loss()      — Masked cross-entropy (assistant tokens only)
  distill_loss()  — Knowledge distillation (KD) for mitosis
  dpo_loss()      — Direct Preference Optimization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


IGNORE_IDX = -100


# ── MRL Loss ──────────────────────────────────────────────────────────────────

def mrl_loss(hidden: torch.Tensor, targets: torch.Tensor,
             embedding_weight: torch.Tensor,
             dims: list[int]) -> torch.Tensor:
    """
    Matryoshka Representation Learning loss.

    Forces the model to pack the most informative features into the
    first `m` dimensions for every m in `dims`. This is what makes
    weight-slicing during mitosis lossless.

    Args:
        hidden:          [B, T, d_model] — final layer hidden states
        targets:         [B, T]          — token IDs (IGNORE_IDX = -100 for padding)
        embedding_weight:[V, d_model]    — weight-tied embedding matrix
        dims:            list of ints    — e.g. [64, 128, 256, 384, 512, 768]

    Returns:
        scalar loss (weighted sum of per-dim CE losses)
    """
    B, T, d_model = hidden.shape
    V = embedding_weight.size(0)
    total_loss = torch.tensor(0.0, device=hidden.device)
    weight_sum = 0.0

    for m in dims:
        assert m <= d_model, f"dim {m} > d_model {d_model}"
        w_m = m / (len(dims) * d_model)  # Importance: larger m = higher weight
        weight_sum += w_m

        # Slice hidden to first m dims, project to vocab via first m cols of E
        h_slice = hidden[:, :, :m]                    # [B, T, m]
        E_slice = embedding_weight[:, :m]             # [V, m]
        logits = h_slice @ E_slice.T                  # [B, T, V]

        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=IGNORE_IDX,
        )
        total_loss = total_loss + w_m * loss

    return total_loss / weight_sum  # Normalize by weight sum


# ── MTP Loss ──────────────────────────────────────────────────────────────────

def mtp_loss(mtp_logits: list[torch.Tensor], targets: torch.Tensor,
             lambdas: list[float]) -> torch.Tensor:
    """
    Multi-Token Prediction loss.

    Each head k predicts token at position t+k+1.
    mtp_logits[k] has shape [B, T-k, V].
    targets has shape [B, T].

    Args:
        mtp_logits: list of K tensors [B, T-k, V]
        targets:    [B, T] token IDs
        lambdas:    [1.0, 0.5, 0.25, 0.125] — decaying weights

    Returns:
        scalar weighted MTP loss
    """
    V = mtp_logits[0].size(-1)
    total_loss = torch.tensor(0.0, device=mtp_logits[0].device)

    for k, (logits_k, lambda_k) in enumerate(zip(mtp_logits, lambdas)):
        B, T_k, _ = logits_k.shape
        # targets for head k: positions k+1 to k+1+T_k
        targets_k = targets[:, k + 1: k + 1 + T_k]  # [B, T_k]
        if targets_k.size(1) < T_k:
            # Sequence too short for this head — skip
            continue
        loss_k = F.cross_entropy(
            logits_k.reshape(B * T_k, V),
            targets_k.reshape(B * T_k),
            ignore_index=IGNORE_IDX,
        )
        total_loss = total_loss + lambda_k * loss_k

    return total_loss / sum(lambdas)


def pretrain_loss(model_output: dict, targets: torch.Tensor,
                  embedding_weight: torch.Tensor,
                  mrl_dims: list[int], mtp_lambdas: list[float],
                  mrl_weight: float = 1.0, mtp_weight: float = 1.0) -> dict:
    """
    Combined pretraining loss: MRL + MTP.
    Returns dict with 'loss', 'mrl_loss', 'mtp_loss'.
    """
    hidden = model_output["hidden"]      # [B, T, d_model]
    mtp_logits = model_output["mtp_logits"]

    l_mrl = mrl_loss(hidden, targets, embedding_weight, mrl_dims)
    l_mtp = mtp_loss(mtp_logits, targets, mtp_lambdas)
    loss = mrl_weight * l_mrl + mtp_weight * l_mtp

    return {"loss": loss, "mrl_loss": l_mrl.item(), "mtp_loss": l_mtp.item()}


# ── SFT Loss ──────────────────────────────────────────────────────────────────

def sft_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Masked cross-entropy on assistant tokens only.
    System + user positions have targets set to IGNORE_IDX=-100.

    Args:
        logits:  [B, T, V]
        targets: [B, T] — assistant tokens are real IDs; others are -100

    Returns:
        scalar loss averaged over assistant tokens only
    """
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=IGNORE_IDX,
    )


# ── Distillation Loss (Mitosis) ───────────────────────────────────────────────

def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 targets: torch.Tensor, alpha: float = 0.15,
                 temperature: float = 3.5) -> torch.Tensor:
    """
    Knowledge Distillation loss for mitosis child training.

    L = α * CE(targets, student) + (1-α) * T² * KL(teacher_soft || student_soft)

    The T² factor compensates for variance reduction at high temperature.

    Args:
        student_logits: [B, T, V] — child model logits
        teacher_logits: [B, T, V] — frozen mother model logits
        targets:        [B, T]    — ground truth token IDs
        alpha:          float     — hard label weight (0.15)
        temperature:    float     — distillation temperature (3.5)
    """
    B, T, V = student_logits.shape

    # Hard label loss
    ce_loss = F.cross_entropy(
        student_logits.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=IGNORE_IDX,
    )

    # Soft label loss (KL divergence)
    with torch.no_grad():
        soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    # KL(teacher || student) = sum(teacher * log(teacher/student))
    kl = F.kl_div(
        soft_student.reshape(B * T, V),
        soft_teacher.reshape(B * T, V).exp(),  # Convert log to prob for target
        reduction="batchmean",
        log_target=False,
    )

    # T² compensation for temperature scaling
    kl_scaled = (temperature ** 2) * kl

    return alpha * ce_loss + (1.0 - alpha) * kl_scaled


# ── DPO Loss ──────────────────────────────────────────────────────────────────

def log_prob_of_completion(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute sum of log probabilities of completion tokens.
    Prompt tokens are masked with IGNORE_IDX.

    Args:
        logits: [B, T, V]
        labels: [B, T]

    Returns: [B] — sum of log probs for completion tokens
    """
    B, T, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
    completion_mask = (labels != IGNORE_IDX).float()  # [B, T]

    # Gather log prob for the actual token at each position
    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.clamp(min=0).unsqueeze(-1)  # clamp to avoid -100 index
    ).squeeze(-1)  # [B, T]

    # Zero out prompt positions and sum
    return (token_log_probs * completion_mask).sum(dim=-1)  # [B]


def dpo_loss(policy_logits_chosen: torch.Tensor,
             policy_logits_rejected: torch.Tensor,
             ref_logits_chosen: torch.Tensor,
             ref_logits_rejected: torch.Tensor,
             labels_chosen: torch.Tensor,
             labels_rejected: torch.Tensor,
             beta: float = 0.1) -> dict:
    """
    Direct Preference Optimization loss.

    L_DPO = -E[log σ(β(log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))]

    Args:
        policy_logits_chosen:   [B, T, V] — π_θ on chosen responses
        policy_logits_rejected: [B, T, V] — π_θ on rejected responses
        ref_logits_chosen:      [B, T, V] — π_ref on chosen (no grad)
        ref_logits_rejected:    [B, T, V] — π_ref on rejected (no grad)
        labels_chosen:          [B, T]    — chosen labels (prompt=-100)
        labels_rejected:        [B, T]    — rejected labels (prompt=-100)
        beta:                   float     — KL regularization (0.1)
    """
    # Log probabilities under policy
    pi_logps_w = log_prob_of_completion(policy_logits_chosen, labels_chosen)
    pi_logps_l = log_prob_of_completion(policy_logits_rejected, labels_rejected)

    # Log probabilities under reference (no gradient)
    with torch.no_grad():
        ref_logps_w = log_prob_of_completion(ref_logits_chosen, labels_chosen)
        ref_logps_l = log_prob_of_completion(ref_logits_rejected, labels_rejected)

    # Implicit rewards
    logratios_w = pi_logps_w - ref_logps_w  # [B]
    logratios_l = pi_logps_l - ref_logps_l  # [B]

    # DPO objective
    loss = -F.logsigmoid(beta * (logratios_w - logratios_l)).mean()

    # Diagnostics
    reward_chosen   = (beta * logratios_w).mean().item()
    reward_rejected = (beta * logratios_l).mean().item()
    reward_margin   = reward_chosen - reward_rejected

    return {
        "loss": loss,
        "reward_chosen": reward_chosen,
        "reward_rejected": reward_rejected,
        "reward_margin": reward_margin,
    }
```

---

## Phase 3 — Pretraining Loop (Day 3–5, runs 72–96h)

### 3.1 `src/data/pretrain_loader.py`

```python
# src/data/pretrain_loader.py
"""
Streaming FineWeb-Edu dataloader.
Never loads the full dataset into RAM — critical for M1.
"""
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import sentencepiece as spm
from pathlib import Path


class FineWebStreamDataset(IterableDataset):
    """
    Streams FineWeb-Edu, tokenizes, and packs tokens into fixed-length chunks.
    Uses a rolling buffer to avoid wasted tokens at document boundaries.
    """
    def __init__(self, tokenizer_path: str, seq_len: int = 1024,
                 bos_id: int = 2, eos_id: int = 3):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)
        self.seq_len = seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __iter__(self):
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        buffer = []
        for sample in dataset:
            text = sample["text"].strip()
            ids = [self.bos_id] + self.sp.Encode(text) + [self.eos_id]
            buffer.extend(ids)
            # Yield complete chunks
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:],  dtype=torch.long)
                yield x, y


def make_pretrain_loader(tokenizer_path: str, seq_len: int = 1024,
                         batch_size: int = 4, num_workers: int = 0) -> DataLoader:
    """
    num_workers=0 is required for streaming datasets on MPS.
    """
    ds = FineWebStreamDataset(tokenizer_path, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=False)
```

### 3.2 `src/training/pretrain.py`

```python
# src/training/pretrain.py
"""
Phase 1: Pretraining loop with MRL + MTP losses.
M1-optimised: FP16 autocast, grad accumulation, no swap.
"""
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from pathlib import Path
import math
import json
import logging
from tqdm import tqdm

from src.config import GenesisConfig
from src.models.genesis import GenesisTransformer
from src.data.pretrain_loader import make_pretrain_loader
from src.training.losses import pretrain_loss
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.device import get_device
from src.utils.memory_monitor import log_memory

log = logging.getLogger(__name__)


def cosine_lr(step: int, warmup: int, max_steps: int,
              lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / warmup
    if step > max_steps:
        return lr_min
    progress = (step - warmup) / (max_steps - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def run_pretrain(
    tokenizer_path: str = "checkpoints/tokenizer/karam_spm_32k.model",
    output_dir: str = "checkpoints/pretrain",
    resume_from: str = None,
    # Training hyperparams (from PRD Section A.2)
    lr_max: float = 6e-4,
    lr_min: float = 6e-5,
    warmup_steps: int = 2_000,
    max_steps: int = 3_100,       # ~100M tokens
    micro_batch: int = 4,
    grad_accum: int = 8,          # Effective batch = 32
    seq_len: int = 1_024,
    grad_clip: float = 1.0,
    checkpoint_interval: int = 500,
    log_interval: int = 10,
    # MRL + MTP weights (from PRD Section 3.5, 3.6)
    mrl_weight: float = 1.0,
    mtp_weight: float = 1.0,
):
    device = get_device()
    log.info(f"Device: {device}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    cfg = GenesisConfig.mother()
    model = GenesisTransformer(cfg).to(device)
    model = model.half()  # FP16 parameters
    log.info(f"Model: {model.count_parameters():,} parameters")

    # ── Optimizer ────────────────────────────────────────────────────────────
    # AdamW with betas=(0.9, 0.95) and weight_decay=0.1
    # Separate param groups: no decay on norm, embedding, biases
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.ndim >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr_max,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=False,  # fused not available on MPS
    )

    # GradScaler for FP16 on MPS
    scaler = GradScaler(device=str(device))

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    if resume_from:
        start_step = load_checkpoint(model, optimizer, resume_from, device)
        log.info(f"Resumed from step {start_step}")

    # ── Dataloader ───────────────────────────────────────────────────────────
    loader = make_pretrain_loader(tokenizer_path, seq_len=seq_len,
                                  batch_size=micro_batch)
    loader_iter = iter(loader)

    # ── Training Loop ────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    running_mrl = 0.0
    running_mtp = 0.0
    tokens_seen = 0

    pbar = tqdm(range(start_step, max_steps), desc="Pretrain", dynamic_ncols=True)
    for step in pbar:
        # Update LR
        lr = cosine_lr(step, warmup_steps, max_steps, lr_max, lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation loop
        accum_loss = 0.0
        for _ in range(grad_accum):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            x, y = x.to(device), y.to(device)

            with autocast(device_type=str(device).split(":")[0], dtype=torch.float16):
                out = model(x, use_mtp=True)
                losses = pretrain_loss(
                    out, y,
                    model.tok_emb.weight,
                    cfg.mrl_dims, cfg.mtp_lambdas,
                    mrl_weight, mtp_weight,
                )
                loss = losses["loss"] / grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            running_mrl += losses["mrl_loss"] / grad_accum
            running_mtp += losses["mtp_loss"] / grad_accum
            tokens_seen += x.numel()

        # Gradient clipping + optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += accum_loss

        # ── Logging ──────────────────────────────────────────────────────────
        if (step + 1) % log_interval == 0:
            avg = running_loss / log_interval
            pbar.set_postfix({
                "loss": f"{avg:.4f}",
                "mrl": f"{running_mrl/log_interval:.4f}",
                "mtp": f"{running_mtp/log_interval:.4f}",
                "lr": f"{lr:.2e}",
                "tok": f"{tokens_seen/1e6:.1f}M",
            })
            running_loss = 0.0
            running_mrl = 0.0
            running_mtp = 0.0
            log_memory(device)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if (step + 1) % checkpoint_interval == 0:
            ckpt_path = output_dir / f"step_{step+1:06d}.pt"
            save_checkpoint(model, optimizer, step + 1, str(ckpt_path), cfg)
            log.info(f"Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    save_checkpoint(model, optimizer, max_steps,
                    str(output_dir / "final.pt"), cfg)
    log.info("✅ Pretraining complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pretrain()
```

---

## Phase 4 — SFT (Day 11–13)

### 4.1 `src/data/sft_loader.py`

```python
# src/data/sft_loader.py
"""
ChatML conversation formatter and dataloader.
Loss mask: system + user tokens get IGNORE_IDX=-100.
Only assistant tokens contribute to the gradient.

ChatML format:
  <bos><|system|>...<|end|>
