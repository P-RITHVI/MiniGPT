from torch import nn
from .attention import CausalSelfAttention

class TransformerBlock(nn.Module):
    """Pre-layer norm block: x + attention(LN(x)), then x + MLP(LN(x))."""
    def __init__(self, d_model, n_heads, dropout, block_size):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model); self.attn=CausalSelfAttention(d_model,n_heads,dropout,block_size)
        self.ln2=nn.LayerNorm(d_model); self.mlp=nn.Sequential(nn.Linear(d_model,4*d_model),nn.GELU(),nn.Linear(4*d_model,d_model),nn.Dropout(dropout))
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); return x+self.mlp(self.ln2(x))
