import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import sentencepiece as spm
import numpy as np


class FineWebStreamingDataset(IterableDataset):
    """
    Streams FineWeb-Edu directly from HuggingFace, tokenizes on the fly,
    and yields packed chunks of (input_ids, targets) of size `seq_len`.
    
    This avoids downloading the massive dataset to the M1's local SSD,
    and creates continuous documents without padding by packing tokens
    and using the EOS token to separate documents.
    """
    def __init__(self, tokenizer_path: str, seq_len: int = 1024, split: str = "train"):
        super().__init__()
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.seq_len = seq_len
        # FineWeb-Edu sample-10BT is ~10 Billion tokens, highly educational.
        # Streaming=True is critical for the M1's storage constraints.
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=split, streaming=True)
        self.eos_id = self.tokenizer.eos_id()

    def __iter__(self):
        buffer = []
        for row in self.dataset:
            text = row["text"]
            # Tokenize and append EOS
            tokens = self.tokenizer.encode(text) + [self.eos_id]
            buffer.extend(tokens)

            # Yield chunks of seq_len + 1 (since we need targets = inputs shifted by 1)
            chunk_size = self.seq_len + 1
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                
                # Convert to uint16 first to save memory before upcasting to long in the dataloader
                chunk_np = np.array(chunk, dtype=np.uint16)
                yield torch.from_numpy(chunk_np).long()


def create_pretrain_dataloader(tokenizer_path: str, batch_size: int, seq_len: int = 1024, num_workers: int = 0) -> DataLoader:
    """
    Creates the DataLoader.
    M1 Unified Memory optimization: keep batch size small (e.g., 4 or 8) 
    and rely on gradient accumulation to hit the target batch size.
    """
    dataset = FineWebStreamingDataset(tokenizer_path, seq_len=seq_len)
    
    def collate_fn(batch):
        # batch is a list of 1D tensors of size seq_len + 1
        stacked = torch.stack(batch) # [B, seq_len + 1]
        input_ids = stacked[:, :-1].contiguous()
        targets = stacked[:, 1:].contiguous()
        return input_ids, targets

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False  # MPS doesn't use standard CUDA pinned memory
    )
