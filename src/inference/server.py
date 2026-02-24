from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import os
import torch
import json

from src.inference.generate import SwarmInferenceSession
from src.swarm.lifecycle import record_node_access, prune_idle_children
from src.config import GenesisConfig

# Load an extremely basic BPE wrapper assuming sentencepiece was trained
class DummyTokenizer:
    """Fallback tokenizer just to make the API run end-to-end without real binaries loaded"""
    def encode(self, text: str): return [1, 2, 3] # mock ids
    def decode(self, ids: list[int]): return " Swarm intelligence."

tokenizer = DummyTokenizer()

app = FastAPI(title="KaramLLM v3 'Genesis' Swarm Interface", version="1.0.0")

# 1. Boot up the dynamic router session (this loads Miniature LM + MLP logic)
try:
    swarm_session = SwarmInferenceSession(tokenizer=tokenizer)
except Exception as e:
    print(f"Server Startup Warning: Router not fully initialized. {e}")
    swarm_session = None


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 50


async def stream_generator(req: ChatRequest):
    """
    Server-Sent Events (SSE) generator for fluid token streaming.
    Simulates the chunked autoregressive forward pass yield natively.
    """
    if not swarm_session:
        yield f"data: {json.dumps({'error': 'Swarm Router offline'})}\n\n"
        return

    # 1. Routing Decision - $O(1)$ constant-cost
    target_node = swarm_session.router.route_prompt(req.prompt)
    
    # 2. Tell the lifecycle manager this node is actively used, blocking Apoptosis prune
    record_node_access(target_node)
    
    # 3. Stream Metadata Header to UI
    yield f"data: {json.dumps({'event': 'routed', 'node': target_node})}\n\n"

    # 4. Generate & Yield Tokens (Simulating yield via the batch API for MVP)
    # Ideally `generate()` is refactored into `generate_stream()` yielding per loop
    try:
        expert = swarm_session._load_node(target_node)
        
        # BPE Encode
        input_ids = swarm_session.tokenizer.encode(req.prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=swarm_session.device)
        
        # We manually break the loop to simulate SSE streaming without rewriting generate.py
        # Real-time token by token delivery
        generated = input_tensor.clone()
        expert.eval()
        
        with torch.no_grad():
            for _ in range(req.max_tokens):
                # Small yield sleep for async concurrency mapping
                await asyncio.sleep(0.01)
                
                idx_cond = generated if generated.size(1) <= expert.config.max_seq_len else generated[:, -expert.config.max_seq_len:]
                outputs = expert(idx_cond, use_mtp=False)
                logits = outputs["logits"][:, -1, :] # Top MTP head

                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)
                
                # Fetch only the single token decoded to send over stream
                new_str = swarm_session.tokenizer.decode([next_token.item()])
                
                # Push Server-Sent Event block
                yield f"data: {json.dumps({'token': new_str})}\n\n"
        
        yield "data: [DONE]\n\n"
            
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/v1/chat/completions")
async def chat_streaming_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main conversational agent interface. Routes immediately to the required node.
    """
    # Push the apoptosis subroutine to background so the UI never waits
    background_tasks.add_task(prune_idle_children)
    
    return StreamingResponse(
        stream_generator(req), 
        media_type="text/event-stream"
    )

@app.get("/health")
def health_check():
    import psutil
    mem = psutil.virtual_memory()
    return {
        "status": "online",
        "active_nodes": list(swarm_session.active_nodes.keys()) if swarm_session else [],
        "system_ram_gb_used": round(mem.used / (1024**3), 2),
    }

# Entry: uvicorn src.inference.server:app --reload --port 8000
