import pytest,torch
from minigpt.config import Config
from minigpt.model import GPTLanguageModel

def test_generation_policy():
    m=GPTLanguageModel(Config(vocab_size=7,block_size=4,d_model=8,n_heads=2,n_layers=1,dropout=0)).eval(); prompt=torch.tensor([[1,2,3,4,5]])
    a=m.generate(prompt,3,greedy=True); b=m.generate(prompt,3,greedy=True); assert a.shape==(1,8) and a[0,:5].tolist()==prompt[0].tolist() and torch.equal(a,b)
    with pytest.raises(ValueError): m.generate(prompt,1,temperature=0)
    with pytest.raises(ValueError): m.generate(prompt,1,top_k=8)
