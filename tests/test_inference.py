import torch
from unittest.mock import MagicMock
from src.inference.generate import generate, SwarmInferenceSession
from src.config import GenesisConfig
from src.models.genesis import GenesisTransformer
from src.swarm.registry import _save_registry

def test_autoregressive_generation_shape():
    """
    Ensures the `generate` function iteratively appends tokens and preserves constraints.
    """
    m_cfg = GenesisConfig.mother()
    mother = GenesisTransformer(m_cfg)
    
    # Batch=1, Length=10 prompt
    mock_prompt = torch.randint(0, m_cfg.vocab_size, (1, 10))
    
    # Request exactly 5 new tokens
    out_tokens = generate(mother, mock_prompt, max_new_tokens=5, temperature=0.0)
    
    assert out_tokens.shape == (1, 15), "Generation failed to append the correct number of tokens."

def test_swarm_router_fallback():
    """
    Ensures that when a prompt has low semantic overlap with any child,
    or no children exist, it falls back to the Mother node.
    """
    # Create empty registry for the test
    _save_registry({"version": "test", "nodes": {}})
    
    # We mock the tokenizer for the session
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.decode.return_value = "Mock response"
    
    session = SwarmInferenceSession(tokenizer=mock_tok, mother_ckpt="dummy_path.pt")
    
    # Should aggressively route to mother if registry is empty
    target = session.router.route_prompt("How do I fix this Python code?")
    assert target == "mother", "Failed to fallback to mother node."

    # Force a mock token ID sequence to test the generation plumbing
    # We intercept the actual loading of the model to avoid FileNotFoundError
    # since we don't have real checkpoints in the test suite.
    dummy_mother = GenesisTransformer(GenesisConfig.mother()).to(session.device)
    session._load_node = MagicMock(return_value=dummy_mother)
    
    response = session.chat("Test prompt", max_tokens=2)
    assert response == "Mock response", "Inference chat orchestration pipeline failed."
