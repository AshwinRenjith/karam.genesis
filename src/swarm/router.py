import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.swarm.registry import get_mini_lm, _load_registry


class SwarmRouter(nn.Module):
    """
    O(1) Constant-Cost Learned Router.
    Instead of calculating routing weights per-token (like Mixtral MoE), 
    we embed the user's initial prompt once using a highly efficient MiniLM, 
    and pass it through a dynamic 2-layer MLP classifier to select the best Node 
    for the entire generation session.
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # MiniLM embedding is exactly 384 dimensions
        self.input_dim = 384
        
        # We always have the Mother node, plus N children
        # The output size (num_classes) changes dynamically as the swarm grows.
        registry = _load_registry()
        self.node_ids = ["mother"] + list(registry["nodes"].keys())
        num_classes = len(self.node_ids)
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # We keep MiniLM outside the PyTorch compute graph to avoid 
        # carrying its gradients during router fine-tuning.
        self.encoder = None

    def _get_encoder(self):
        if self.encoder is None:
            self.encoder = get_mini_lm()
        return self.encoder

    def forward(self, semantic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Expects semantic_embeddings of shape [B, 384]
        Returns raw class logits [B, num_classes]
        """
        x = F.relu(self.fc1(semantic_embeddings))
        logits = self.fc2(x)
        return logits

    @torch.no_grad()
    def route_prompt(self, user_prompt: str, threshold: float = 0.55) -> str:
        """
        Inference API: Takes a raw string, embeds it, and returns the target node_id.
        If the highest child confidence is below the threshold, it falls back to Mother.
        """
        encoder = self._get_encoder()
        # [1, 384] numpy array -> tensor
        embedding_np = encoder.encode([user_prompt])
        embedding = torch.from_numpy(embedding_np).float().to(self.fc1.weight.device)
        
        logits = self.forward(embedding)
        probs = F.softmax(logits, dim=-1)[0] # [num_classes]
        
        max_prob, max_idx = torch.max(probs, dim=0)
        
        # Index 0 is always Mother
        if max_idx == 0:
            return "mother"
            
        # If the top child's probability is below our absolute threshold,
        # the router is uncertain. Fall back to the general-purpose Mother.
        if max_prob < threshold:
            return "mother"
            
        return self.node_ids[max_idx.item()]


def initialize_router_from_registry(save_path: str = "checkpoints/swarm/router.pt"):
    """
    Initializes the router's weights.
    For a newly spawned child, we could set the specific output logit bias 
    to closely match the computed cluster centroid via cosine similarity initialization.
    """
    router = SwarmRouter()
    
    registry = _load_registry()
    node_keys = list(registry["nodes"].keys())
    
    # If no children exist, we don't really need to train anything, it just routes to Mother
    if not node_keys:
        print("💡 No children in registry. Router defaults exclusively to Mother.")
        torch.save(router.state_dict(), save_path)
        return router

    print(f"🧠 Initializing Router for {len(node_keys)} child nodes + Mother...")
    
    # Heuristic Initialization:
    # We can pre-bias the classification weights using the exact domain centroids
    # computed during the POST-SPAWN phase so the router is mostly accurate at step 0.
    with torch.no_grad():
        centroids = []
        for kid in node_keys:
            cent = registry["nodes"][kid].get("centroid_384d")
            centroids.append(cent)
        
        # centroids is [num_children, 384]
        # Mother acts as a catch-all, we can treat her "centroid" as a zero vector 
        # or something orthogonal to the highly specific domains.
        c_tensor = torch.tensor(centroids, dtype=torch.float32) # [N, 384]
        
        # We roughly map fc1 * fc2 to be similar to a dot product against centroids
        # For an exact zero-shot start, we could bypass the MLP and just do cosine 
        # similarity, but the MLP allows continuous active learning.
        pass # Keeping MLP standard initialization for now, fine-tuning solves it fast.
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(router.state_dict(), save_path)
    return router
