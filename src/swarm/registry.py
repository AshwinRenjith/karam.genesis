import os
import json
import torch
from sentence_transformers import SentenceTransformer

# Registry path
REGISTRY_PATH = "checkpoints/swarm/node_registry.json"

def get_mini_lm():
    """Returns a lightweight MiniLM model for domain centroid computation."""
    # all-MiniLM-L6-v2 is extremely fast on M1 and produces 384d semantic vectors
    return SentenceTransformer('all-MiniLM-L6-v2')

def _load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r') as f:
            return json.load(f)
    return {"version": "genesis_1.0", "nodes": {}}

def _save_registry(registry: dict):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)

def compute_domain_centroid(corpus_texts: list[str]) -> list[float]:
    """
    Computes Semantic Centroid of a domain corpus for routing.
    Embeds documents via MiniLM and takes the mean.
    Returns a standard python list to be JSON serializable.
    """
    model = get_mini_lm()
    print(f"🧠 Computing centroid over {len(corpus_texts)} documents...")
    
    # Encode returns numpy array [N, 384]
    embeddings = model.encode(corpus_texts, show_progress_bar=True)
    
    # Calculate the mean vector
    centroid = embeddings.mean(axis=0)
    return centroid.tolist()

def register_node(node_id: str, config_dict: dict, corpus_texts: list[str]):
    """
    POST-SPAWN: Step 1 & 2.
    Computes domain centroid and registers the freshly spawned Child node.
    """
    print(f"📝 Registering Node: {node_id}")
    registry = _load_registry()
    
    centroid = compute_domain_centroid(corpus_texts)
    
    registry["nodes"][node_id] = {
        "status": "active",
        "d_model": config_dict.get("d_model", 512),
        "params_m": 66.3, # Roughly based on the scaled math
        "centroid_384d": centroid,
    }
    
    _save_registry(registry)
    print(f"✅ Node '{node_id}' formally registered in swarm.")
