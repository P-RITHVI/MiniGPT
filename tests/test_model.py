import torch
import pytest
from minigpt.config import Config
from minigpt.model import GPTLanguageModel

def tiny(): return Config(vocab_size=12,block_size=8,d_model=16,n_heads=4,n_layers=2,dropout=0)
def test_forward_shapes_loss():
    m=GPTLanguageModel(tiny()); x=torch.randint(0,12,(3,7)); logits,loss=m(x,x); assert logits.shape==(3,7,12); assert loss.ndim==0 and torch.isfinite(loss)
def test_invalid_config_context():
    with pytest.raises(ValueError): GPTLanguageModel(Config(vocab_size=9,d_model=10,n_heads=3))
    with pytest.raises(ValueError): GPTLanguageModel(tiny())(torch.zeros((1,9),dtype=torch.long))
