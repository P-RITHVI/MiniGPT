import torch
from minigpt.checkpoint import save_checkpoint,load_checkpoint
from minigpt.config import Config
from minigpt.model import GPTLanguageModel
from minigpt.tokenizer import CharTokenizer

def test_roundtrip(tmp_path):
    c=Config(vocab_size=4,block_size=5,d_model=8,n_heads=2,n_layers=1,dropout=0); t=CharTokenizer.train("abc"); c.vocab_size=t.vocab_size; m=GPTLanguageModel(c).eval(); p=tmp_path/"m.pt"; save_checkpoint(p,m,t,c)
    loaded,t2,_,_=load_checkpoint(p); x=torch.tensor([[1,2]]); assert t2.itos==t.itos; assert torch.allclose(m(x)[0],loaded(x)[0])
