import os
import math
import torch
import torch.nn as nn
from typing import Optional

from src.config import GenesisConfig
from src.models.genesis import GenesisTransformer
from src.data.pretrain_loader import create_pretrain_dataloader
from src.training.losses import pretrain_loss

# MPS (Metal Performance Shaders) specific setup for M1
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def cosine_lr_schedule(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    """Cosine learning rate with linear warmup."""
    if step < warmup_steps:
        return peak_lr * (step / warmup_steps)
    
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    decay_ratio = min(1.0, decay_ratio)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def pretrain(
    tokenizer_path: str = "checkpoints/tokenizer/karam_spm_32k.model",
    save_dir: str = "checkpoints/pretrain",
    total_steps: int = 100_000,
    batch_size: int = 1,              # Micro batch size REDUCED to prevent MPS OOM
    grad_accum_steps: int = 32,       # Effective batch size = 32
):
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()
    print(f"🚀 Starting Genesis Pretraining on {device}")

    # 1. Initialize Canonical Mother Configuration
    config = GenesisConfig.mother()
    model = GenesisTransformer(config).to(device)
    
    # Enable Half Precision (FP16) or BF16 depending on MPS support
    # For M1, standard FP32 or mixed precision is fine. We explicitly cast in the loop if needed.
    
    print(f"Model parameters: {model.count_parameters():,} (~{model.count_parameters()/1e6:.1f}M)")

    # 2. Setup Optimizer (Simulating AdamW 8-bit requirements)
    # Note: true 8-bit Adam requires extreme custom kernels for MPS.
    # We use standard AdamW here; for true 8-bit on M1, bitsandbytes-mlx is required.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6e-4,                      # Will be overwritten by scheduler immediately
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1
    )

    # 3. Setup DataLoader
    train_loader = create_pretrain_dataloader(
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        seq_len=config.max_seq_len
    )
    data_iter = iter(train_loader)

    # 4. Hyperparameters
    warmup_steps = 2000
    peak_lr = 6e-4
    min_lr = 6e-5
    
    # Loss hyperparams matching PRD
    mrl_dims = [64, 128, 256, 384, 512, 768]
    mtp_lambdas = [1.0, 0.5, 0.25, 0.125]
    
    model.train()
    step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    print("Entering Training Loop...")
    while step < total_steps:
        # Get learning rate
        lr = cosine_lr_schedule(step, total_steps, warmup_steps, peak_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        step_loss = 0.0
        
        # Gradient Accumulation Loop
        for micro_step in range(grad_accum_steps):
            try:
                input_ids, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids, targets = next(data_iter)

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward pass (enable MTP)
            # Mixed precision is tricky natively on MPS without autocast support 
            # for all ops, so we train in full precision or cast manually.
            outputs = model(input_ids, use_mtp=True)

            # Complex Loss: MRL + MTP
            loss_dict = pretrain_loss(
                model_output=outputs,
                targets=targets,
                embedding_weight=model.tok_emb.weight,
                mrl_dims=mrl_dims,
                mtp_lambdas=mtp_lambdas
            )
            
            loss = loss_dict["loss"] / grad_accum_steps
            loss.backward()
            
            step_loss += loss.item()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer Step
        optimizer.step()
        optimizer.zero_grad()
        
        running_loss += step_loss
        
        # Logging
        if step % 10 == 0 or step == 0:
            avg_loss = running_loss / 10 if step > 0 else step_loss
            print(f"Step {step:05d} | LR {lr:.2e} | Loss {avg_loss:.4f} | MRL {loss_dict['mrl_loss']:.4f} | MTP {loss_dict['mtp_loss']:.4f}")
            running_loss = 0.0

        # Checkpointing
        if step > 0 and step % 500 == 0:
            ckpt_path = os.path.join(save_dir, f"genesis_step_{step:06d}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': step_loss,
            }, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

        step += 1

    print("✅ Pretraining Complete!")

if __name__ == "__main__":
    pretrain(total_steps=10) # Run 10 steps as a smoke test by default
