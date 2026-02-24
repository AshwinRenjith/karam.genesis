import os
import torch
import pytest
from src.config import GenesisConfig
from src.models.genesis import GenesisTransformer
from src.data.pretrain_loader import create_pretrain_dataloader
from src.training.losses import pretrain_loss

@pytest.fixture
def mock_tokenizer(tmp_path):
    """Creates a tiny dummy tokenizer file just for testing the data loader instantiation."""
    import sentencepiece as spm
    
    # We just need any valid SPM model to load.
    # We can write a tiny valid sentencepiece file or just mock the dataset entirely.
    # Let's mock the streaming dataset to return random tensors instead to avoid networking in tests
    pass

class MockDataloader:
    def __init__(self, batch_size, seq_len, vocab_size):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= 2:
            raise StopIteration
        self.count += 1
        inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        return inputs, targets


def test_pretrain_integration():
    """
    Integration test: Runs a fake mini-batch through the GenesisTransformer,
    computes the complex MRL+MTP loss, and verifies backward pass computes without crashing.
    """
    config = GenesisConfig.mother()
    config.n_layers = 2 # make it fast
    config.max_seq_len = 128
    
    model = GenesisTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    loader = MockDataloader(batch_size=2, seq_len=128, vocab_size=config.vocab_size)
    
    mrl_dims = [64, 128, 256, 384, 512, 768]
    mtp_lambdas = [1.0, 0.5, 0.25, 0.125]
    
    model.train()
    optimizer.zero_grad()
    
    input_ids, targets = next(iter(loader))
    
    outputs = model(input_ids, use_mtp=True)
    
    loss_dict = pretrain_loss(
        model_output=outputs,
        targets=targets,
        embedding_weight=model.tok_emb.weight,
        mrl_dims=mrl_dims,
        mtp_lambdas=mtp_lambdas
    )
    
    loss = loss_dict["loss"]
    
    assert loss.item() > 0
    assert loss_dict["mrl_loss"] > 0
    assert loss_dict["mtp_loss"] > 0
    
    loss.backward()
    
    # Check if gradients flowed to a couple of random components
    assert getattr(model.tok_emb.weight, 'grad', None) is not None
    assert getattr(model.layers[0].attn.wq.weight, 'grad', None) is not None
    assert getattr(model.mtp_head.heads[0].w1.weight, 'grad', None) is not None
    
    optimizer.step()
