import torch
import torch.nn.functional as F
from src.config import GenesisConfig
from src.models.genesis import GenesisTransformer
from src.swarm.router import SwarmRouter


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def generate(
    model: GenesisTransformer,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
) -> torch.Tensor:
    """
    Standard autoregressive generation loop.
    For Genesis, we only use the primary head (k=1) of the MTP block.
    """
    device = prompt_tokens.device
    model.eval()
    
    generated = prompt_tokens.clone()
    
    for _ in range(max_new_tokens):
        # Truncate context to window size if needed
        # Genesis local window is 256, but global attention can look back up to max_seq_len
        idx_cond = generated if generated.size(1) <= model.config.max_seq_len else generated[:, -model.config.max_seq_len:]
        
        # Forward pass (only taking primary logits)
        outputs = model(idx_cond, use_mtp=False)
        logits = outputs["logits"] # [B, SeqLen, Vocab]
        
        # Get the logits for the *last* token in the sequence
        next_token_logits = logits[:, -1, :] # [B, Vocab]
        
        # Temperature scaling
        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            
            # Top-K filtering
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, -1, None]] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
        generated = torch.cat((generated, next_token), dim=1)
        
    return generated


class SwarmInferenceSession:
    """
    Manages the lifecycle of inference. 
    Loads the Router, receives a prompt, routes it, dynamically loads 
    the selected expert node, and generates the response.
    """
    def __init__(self, tokenizer, mother_ckpt: str = "checkpoints/swarm/mother_final.pt"):
        self.device = get_device()
        self.tokenizer = tokenizer
        
        print("🧠 Bootstrapping Swarm Router...")
        self.router = SwarmRouter().to(self.device)
        self.router.eval()
        
        self.mother_ckpt = mother_ckpt
        
        # Cache for loaded models to avoid constant desk reads
        # (Since children are INT4 or FP16 50M, we can keep many in RAM, 
        # but for this MVP script we just cache them as they are called)
        self.active_nodes = {}
        
    def _load_node(self, node_id: str) -> GenesisTransformer:
        if node_id in self.active_nodes:
            return self.active_nodes[node_id]
            
        print(f"🔄 Loading node '{node_id}' into memory...")
        
        if node_id == "mother":
            config = GenesisConfig.mother()
            ckpt_path = self.mother_ckpt
        else:
            config = GenesisConfig.child(node_id)
            ckpt_path = f"checkpoints/swarm/{node_id}_final.pt"
            
        model = GenesisTransformer(config).to(self.device)
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
        else:
            print(f"⚠️ Warning: Checkpoint for {node_id} not found. Operating with random weights!")
            
        model.eval()
        self.active_nodes[node_id] = model
        return model

    def chat(self, user_prompt: str, max_tokens: int = 150) -> str:
        # 1. Routing
        target_node = self.router.route_prompt(user_prompt)
        print(f"🎯 Router decision: Directed to '{target_node}'")
        
        # 2. Load Model
        expert = self._load_node(target_node)
        
        # 3. Tokenize
        # Note: BPE tokenizer expects standard Python strings
        input_ids = self.tokenizer.encode(user_prompt)
        # SentencePiece return list of ints. Convert to tensor.
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # 4. Generate
        print(f"⚡ Generating response from {target_node}...")
        out_ids = generate(expert, input_tensor, max_new_tokens=max_tokens)
        
        # 5. Decode
        # Slicing from input_tensor length to get *only* the new tokens
        new_ids = out_ids[0][input_tensor.size(1):].tolist()
        response = self.tokenizer.decode(new_ids)
        
        return response
