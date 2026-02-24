import os
import torch
from src.config import GenesisConfig
from src.swarm.mitosis import spawn_expert
from src.training.losses import distill_loss


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def distill_child(
    mother_ckpt_path: str,
    domain_node_id: str,
    save_dir: str = "checkpoints/swarm",
    total_steps: int = 10_000,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
):
    """
    Knowledge Distillation training loop for a newly spawned Child node.
    The Mother acts as the frozen teacher to preserve general knowledge and structure,
    while the Child learns from the domain-specific data using hard labels + KD.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()
    print(f"🧬 Starting Knowledge Distillation for Child '{domain_node_id}' on {device}")

    # 1. Spawn Child (Performs Mitosis internally)
    child = spawn_expert(mother_ckpt_path, domain_node_id).to(device)
    
    # 2. Load Frozen Mother (Teacher)
    mother_config = GenesisConfig.mother()
    mother = spawn_expert(mother_ckpt_path, "teacher_placeholder") # Use same loader logic
    mother = mother.to(device)
    mother.eval() # Freeze teacher
    for param in mother.parameters():
        param.requires_grad = False

    # 3. Optimizer
    # Note: KD is mostly FP16 + FP32 optimizer (Memory is low since child is 50M)
    optimizer = torch.optim.AdamW(
        child.parameters(),
        lr=1e-4,                      
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01
    )

    # 4. Dummy DataLoader for the script structure
    # In reality this uses the same FineWeb loader but pointed to the domain corpus
    def dummy_data_gen():
        while True:
            # 1024 seq len
            inputs = torch.randint(0, 32000, (batch_size, 1024), device=device)
            targets = torch.randint(0, 32000, (batch_size, 1024), device=device)
            yield inputs, targets
            
    data_iter = dummy_data_gen()

    child.train()
    step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    print("Entering Distillation Loop...")
    while step < total_steps:
        step_loss = 0.0
        
        for micro_step in range(grad_accum_steps):
            input_ids, targets = next(data_iter)

            # Teacher Forward (No Gradients)
            with torch.no_grad():
                teacher_outputs = mother(input_ids)
                teacher_logits = teacher_outputs["logits"]

            # Student Forward
            student_outputs = child(input_ids)
            student_logits = student_outputs["logits"]

            # KD Loss
            loss = distill_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=targets,
                alpha=0.15,       # From PRD hard-label weight
                temperature=3.5   # From PRD softening temperature
            )
            
            loss = loss / grad_accum_steps
            loss.backward()
            step_loss += loss.item()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(child.parameters(), max_norm=1.0)
        
        # Optimizer Step
        optimizer.step()
        optimizer.zero_grad()
        
        running_loss += step_loss
        
        # Logging
        if step % 10 == 0 or step == 0:
            avg_loss = running_loss / 10 if step > 0 else step_loss
            print(f"Distill Step {step:05d} | Loss {avg_loss:.4f}")
            running_loss = 0.0

        step += 1

    # Save final child spawn
    ckpt_path = os.path.join(save_dir, f"{domain_node_id}_final.pt")
    torch.save({
        'node_id': domain_node_id,
        'config': child.config.__dict__,
        'model_state_dict': child.state_dict()
    }, ckpt_path)
    print(f"✅ Child '{domain_node_id}' Distillation Complete! Saved to {ckpt_path}")


if __name__ == "__main__":
    # Smoke test
    distill_child("dummy", "finance_child", total_steps=5)
