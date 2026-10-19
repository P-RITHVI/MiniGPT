import math
import torch
from torch import nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, block_size=256):
        super().__init__()
        if d_model % n_heads: raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model); self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout); self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)), persistent=False)
    def forward(self, x):
        b,t,c = x.shape
        if t > self.causal_mask.size(0): raise ValueError("Sequence exceeds attention context")
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        shape = (b,t,self.n_heads,self.head_dim)
        q,k,v = [z.view(shape).transpose(1,2) for z in (q,k,v)]
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:t,:t], torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1,2).contiguous().view(b,t,c)
        return self.resid_dropout(self.proj(out))
