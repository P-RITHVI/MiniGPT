import torch
from minigpt.config import Config
from minigpt.model import GPTLanguageModel

def test_backward_and_update():
    m=GPTLanguageModel(Config(vocab_size=8,block_size=5,d_model=8,n_heads=2,n_layers=1,dropout=0)); x=torch.randint(0,8,(2,5)); before=m.token_embedding.weight.detach().clone()
    _,loss=m(x,x); loss.backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()); torch.optim.AdamW(m.parameters(),lr=.01).step(); assert not torch.equal(before,m.token_embedding.weight)
