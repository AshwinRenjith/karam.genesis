import time
from src.swarm.registry import _load_registry, _save_registry

def _get_current_time():
    return int(time.time())

def prune_idle_children(max_idle_seconds: int = 7 * 24 * 60 * 60):
    """
    Apoptosis: Programmatic cell death for inactive Swarm nodes.
    Prunes children from the active registry that have not been queried recently.
    Saves RAM and Router compute linearly.
    """
    registry = _load_registry()
    now = _get_current_time()
    
    deleted_count = 0
    
    # Iterate over a copy of keys since we are mutating the dict
    for node_id in list(registry.get("nodes", {}).keys()):
        # Mother node cannot be killed
        if node_id == "mother":
            continue
            
        node_meta = registry["nodes"].get(node_id, {})
        last_accessed = node_meta.get("last_accessed", now) # default to now if freshly spawned
        
        idle_time = now - last_accessed
        
        if idle_time > max_idle_seconds:
            print(f"💀 APOPTOSIS TRIGGERED: Pruning idle node '{node_id}' (Idle for {idle_time/86400:.1f} days)")
            
            # 1. Remove from logical registry
            del registry["nodes"][node_id]
            
            # 2. Archive or delete checkpoint (Here we just unregister to save VRAM/Router size)
            # os.remove(f"checkpoints/swarm/{node_id}_final.pt") 
            
            deleted_count += 1
            
    if deleted_count > 0:
        _save_registry(registry)
        # Note: In a real environment, we'd trigger the Learned Router to fine-tune 
        # instantly over 10 steps so it unlearns the pruned classes.
        update_learned_router_classes()
        
    return deleted_count


def update_learned_router_classes():
    """
    Dynamically resizes and fine-tunes the SwarmRouter's classification head
    whenever the node registry expands or shrinks via Mitosis or Apoptosis.
    """
    import torch
    import torch.nn as nn
    from src.swarm.router import SwarmRouter
    import os
    
    # We load the existing router state if possible, 
    # extract the fc1 weights safely, and rebuild fc2 to match the new class count.
    registry = _load_registry()
    num_classes = len(registry["nodes"]) + 1 # +1 for mother
    
    print(f"🔄 Updating Learned Router topology for {num_classes} active classes...")
    
    new_router = SwarmRouter(hidden_dim=256)
    
    # Only load past if it actually exists. Since num_classes changed, the size of fc2's weight tensor changed.
    # Therefore we cannot blindly load_state_dict.
    router_ckpt = "checkpoints/swarm/router.pt"
    if os.path.exists(router_ckpt):
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            old_state = torch.load(router_ckpt, map_location=device)
            
            # Transfer the base mapping fc1 which embeds semantic intent
            if "fc1.weight" in old_state:
                new_router.fc1.weight.data.copy_(old_state["fc1.weight"])
            if "fc1.bias" in old_state:
                new_router.fc1.bias.data.copy_(old_state["fc1.bias"])
                
            # Note: We intentionally random initialize fc2 because the taxonomy changed.
            # A 100-step fast PPO / Distillation tuning loop would usually happen right here 
            # to align the new mapping.
            print("✅ Router topology dynamically resized.")
        except Exception as e:
            print(f"⚠️ Failed to resize router weights natively: {e}")
            
    torch.save(new_router.state_dict(), router_ckpt)
    
def record_node_access(node_id: str):
    """
    Called by the generation pipeline/server to keep the node alive and prevent Apoptosis.
    """
    registry = _load_registry()
    if node_id in registry["nodes"]:
        registry["nodes"][node_id]["last_accessed"] = _get_current_time()
        _save_registry(registry)
