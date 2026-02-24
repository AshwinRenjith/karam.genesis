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


IGNORE_IDX = -100


# ── MRL Loss ──────────────────────────────────────────────────────────────────

def mrl_loss(hidden: torch.Tensor, targets: torch.Tensor,
             embedding_weight: torch.Tensor,
             dims: list[int]) -> torch.Tensor:
    """
    Matryoshka Representation Learning loss.

    Forces the model to pack the most informative features into the
    first `m` dimensions for every m in `dims`. This makes weight-slicing lossless.

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

        # Slice hidden to first m dims
        # MPS Backend crashes/hangs on transposed non-contiguous matmuls.
        # We MUST force contiguous memory before doing the linear projection.
        h_slice = hidden[:, :, :m].contiguous()       # [B, T, m]
        E_slice = embedding_weight[:, :m].contiguous() # [V, m]
        
        # F.linear(input, weight) equates to input @ weight.T
        logits = F.linear(h_slice, E_slice)           # [B, T, V]

        loss = F.cross_entropy(
            logits.view(B * T, V),
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
        # target token for prediction at index t by head k is at t + k + 1
        # Therefore, targets for head k = targets[:, k+1 : k+1+T_k]
        targets_k = targets[:, k + 1: k + 1 + T_k]
        
        # If sequence is too short to have targets for this k, skip
        if targets_k.size(1) == 0:
            continue
            
        # Logits might be slightly longer than available targets at the very end of the sequence.
        # We must truncate logits_k to match the actually available targets_k
        valid_len = targets_k.size(1)
        logits_k = logits_k[:, :valid_len, :]

        loss_k = F.cross_entropy(
            logits_k.contiguous().view(B * valid_len, V),
            targets_k.contiguous().view(B * valid_len),
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
    """
    B, T, V = student_logits.shape

    ce_loss = F.cross_entropy(
        student_logits.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=IGNORE_IDX,
    )

    with torch.no_grad():
        soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        soft_student.reshape(B * T, V),
        soft_teacher.reshape(B * T, V).exp(),
        reduction="batchmean",
        log_target=False,
    )

    kl_scaled = (temperature ** 2) * kl
    return alpha * ce_loss + (1.0 - alpha) * kl_scaled


# ── DPO Loss ──────────────────────────────────────────────────────────────────

def log_prob_of_completion(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log probabilities of completion tokens (ignoring -100)."""
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
    completion_mask = (labels != IGNORE_IDX).float()  # [B, T]

    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)  # [B, T]

    return (token_log_probs * completion_mask).sum(dim=-1)  # [B]


def dpo_loss(policy_logits_chosen: torch.Tensor,
             policy_logits_rejected: torch.Tensor,
             ref_logits_chosen: torch.Tensor,
             ref_logits_rejected: torch.Tensor,
             labels_chosen: torch.Tensor,
             labels_rejected: torch.Tensor,
             beta: float = 0.1) -> dict:
    """Direct Preference Optimization loss."""
    pi_logps_w = log_prob_of_completion(policy_logits_chosen, labels_chosen)
    pi_logps_l = log_prob_of_completion(policy_logits_rejected, labels_rejected)

    with torch.no_grad():
        ref_logps_w = log_prob_of_completion(ref_logits_chosen, labels_chosen)
        ref_logps_l = log_prob_of_completion(ref_logits_rejected, labels_rejected)

    logratios_w = pi_logps_w - ref_logps_w  # [B]
    logratios_l = pi_logps_l - ref_logps_l  # [B]

    loss = -F.logsigmoid(beta * (logratios_w - logratios_l)).mean()

    return {
        "loss": loss,
        "reward_chosen": (beta * logratios_w).mean().item(),
        "reward_rejected": (beta * logratios_l).mean().item(),
        "reward_margin": (beta * logratios_w).mean().item() - (beta * logratios_l).mean().item(),
    }
