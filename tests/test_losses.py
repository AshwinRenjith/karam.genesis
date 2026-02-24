import torch
import torch.nn.functional as F
from src.training.losses import mrl_loss, mtp_loss, distill_loss, IGNORE_IDX


def test_mrl_loss():
    B, T, d_model = 2, 8, 128
    V = 1000
    dims = [32, 64, 128]
    hidden = torch.randn(B, T, d_model)
    targets = torch.randint(0, V, (B, T))
    # padding simulation
    targets[0, 5:] = IGNORE_IDX
    embedding_weight = torch.randn(V, d_model)

    loss = mrl_loss(hidden, targets, embedding_weight, dims)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"


def test_mtp_loss():
    B, T, V = 2, 8, 1000
    k_vals = 4
    lambdas = [1.0, 0.5, 0.25, 0.125]
    mtp_logits = []
    
    for k in range(k_vals):
        T_k = T - k - 1
        if T_k > 0:
            mtp_logits.append(torch.randn(B, T_k, V))

    targets = torch.randint(0, V, (B, T))
    loss = mtp_loss(mtp_logits, targets, lambdas[:len(mtp_logits)])
    
    assert loss.dim() == 0
    assert loss.item() > 0


def test_distill_loss():
    B, T, V = 2, 8, 1000
    student_logits = torch.randn(B, T, V)
    teacher_logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    
    loss = distill_loss(student_logits, teacher_logits, targets, alpha=0.15)
    assert loss.dim() == 0
    assert loss.item() > 0

    # Test exact matching teacher/student
    loss_exact = distill_loss(teacher_logits, teacher_logits, targets, alpha=0.15)
    # the KL divergence part should be 0, so it's just the CE loss * alpha
    ce_loss = F.cross_entropy(teacher_logits.reshape(-1, V), targets.reshape(-1), ignore_index=IGNORE_IDX)
    assert torch.isclose(loss_exact, ce_loss * 0.15, atol=1e-5)
