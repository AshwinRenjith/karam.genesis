import os
import torch
import torch.nn.functional as F
from src.config import GenesisConfig
from src.models.genesis import GenesisTransformer
from src.swarm.registry import _load_registry, _save_registry


def load_checkpoint_weights(ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    return state['model_state_dict'] if 'model_state_dict' in state else state


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD: float = 0.9995) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) between two tensors v0 and v1.
    This preserves the geometric properties of high-dimensional neural network weights
    far better than simple arithmetic averaging (v0 + v1)/2.
    """
    # Create copies and flatten
    v0_copy = v0.clone().flatten()
    v1_copy = v1.clone().flatten()

    # Normalize vectors
    v0_norm = F.normalize(v0_copy, p=2, dim=0)
    v1_norm = F.normalize(v1_copy, p=2, dim=0)

    # Compute dot product
    dot = torch.sum(v0_norm * v1_norm)

    # If the inputs are colinear (almost identical), fallback to linear interpolation to avoid divide-by-zero
    if torch.abs(dot) > DOT_THRESHOLD:
        res = (1.0 - t) * v0 + t * v1
        return res

    # Calculate angle theta between vectors
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    theta_t = theta_0 * t

    # Calculate orthogonal projection
    v2 = v1_copy - v0_copy * dot
    v2_norm = F.normalize(v2, p=2, dim=0)

    # Interpolate
    res_flat = (v0_copy * torch.cos(theta_t)) + (v2_norm * torch.sin(theta_t) * torch.norm(v0_copy))

    # Reshape back to original shape
    return res_flat.view(v0.shape)


def perform_slerp_fusion(
    node_a_id: str,
    node_b_id: str,
    fused_node_id: str,
    t: float = 0.5,
    checkpoints_dir: str = "checkpoints/swarm"
) -> str:
    """
    Fuses two experts (Child nodes) into a single unified expert using SLERP.
    Allows KaramLLM Genesis to consolidate its expert space optimally and prevent bloat.
    """
    device = get_device()
    print(f"🧬 Initiating SLERP Fusion (t={t}): [{node_a_id}] + [{node_b_id}] -> [{fused_node_id}]")

    registry = _load_registry()
    if node_a_id not in registry["nodes"]:
        raise ValueError(f"Node {node_a_id} not found in registry.")
    if node_b_id not in registry["nodes"]:
        raise ValueError(f"Node {node_b_id} not found in registry.")

    # Load weights
    path_a = os.path.join(checkpoints_dir, f"{node_a_id}_final.pt")
    path_b = os.path.join(checkpoints_dir, f"{node_b_id}_final.pt")
    
    state_a = load_checkpoint_weights(path_a, device)
    state_b = load_checkpoint_weights(path_b, device)

    # Initialize empty child model
    c_config = GenesisConfig.child(node_id=fused_node_id)
    fused_model = GenesisTransformer(c_config).to(device)
    fused_state = fused_model.state_dict()

    print("Executing Spherical Linear Interpolation across all parameter manifolds...")
    with torch.no_grad():
        for key in state_a.keys():
            if key in state_b and key in fused_state:
                # Skip invariant buffers that don't need SLERP (like RoPE frequencies)
                if not isinstance(state_a[key], torch.Tensor) or state_a[key].dtype not in [torch.float16, torch.float32]:
                    fused_state[key].copy_(state_a[key])
                    continue
                
                tensor_a = state_a[key].float()
                tensor_b = state_b[key].float()

                # Perform actual SLERP
                fused_tensor = slerp(t, tensor_a, tensor_b)
                
                # Copy back
                fused_state[key].copy_(fused_tensor.type_as(fused_state[key]))

    fused_model.load_state_dict(fused_state)

    # Calculate new centroid (Linear interpolation of semantic spaces is safe theoretically)
    cent_a = torch.tensor(registry["nodes"][node_a_id]["centroid_384d"])
    cent_b = torch.tensor(registry["nodes"][node_b_id]["centroid_384d"])
    fused_centroid = ((1.0 - t) * cent_a + t * cent_b).tolist()

    # Save Registry
    registry["nodes"][fused_node_id] = {
        "status": "active",
        "d_model": c_config.d_model,
        "params_m": 66.3,
        "centroid_384d": fused_centroid,
        "fused_from": [node_a_id, node_b_id]
    }
    _save_registry(registry)

    # Save Checkpoint
    save_path = os.path.join(checkpoints_dir, f"{fused_node_id}_final.pt")
    torch.save({
        'node_id': fused_node_id,
        'config': c_config.__dict__,
        'model_state_dict': fused_model.state_dict()
    }, save_path)

    print(f"✅ SLERP Fusion Complete. Manifold '{fused_node_id}' safely encoded.")
    return save_path
