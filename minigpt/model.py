import torch
from torch import nn
from .transformer_block import TransformerBlock

class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__(); config.validate()
        self.config=config
        self.token_embedding=nn.Embedding(config.vocab_size,config.d_model)
        self.position_embedding=nn.Embedding(config.block_size,config.d_model)
        self.drop=nn.Dropout(config.dropout)
        self.blocks=nn.ModuleList([TransformerBlock(config.d_model,config.n_heads,config.dropout,config.block_size) for _ in range(config.n_layers)])
        self.ln_f=nn.LayerNorm(config.d_model); self.head=nn.Linear(config.d_model,config.vocab_size,bias=False)
        self.head.weight=self.token_embedding.weight
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(module):
        if isinstance(module,nn.Linear): nn.init.normal_(module.weight,mean=0.0,std=.02); nn.init.zeros_(module.bias) if module.bias is not None else None
        if isinstance(module,nn.Embedding): nn.init.normal_(module.weight,mean=0.0,std=.02)
    def forward(self,idx,targets=None):
        b,t=idx.shape
        if t>self.config.block_size: raise ValueError("Input exceeds configured block_size")
        pos=torch.arange(t,device=idx.device)
        x=self.drop(self.token_embedding(idx)+self.position_embedding(pos))
        for block in self.blocks: x=block(x)
        logits=self.head(self.ln_f(x)); loss=None
        if targets is not None: loss=nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss
    @torch.inference_mode()
    def generate(self,idx,max_new_tokens,temperature=1.0,top_k=None,greedy=False):
        if max_new_tokens<0: raise ValueError("max_new_tokens must be nonnegative")
        if temperature<=0: raise ValueError("temperature must be positive")
        if top_k is not None and not 1<=top_k<=self.config.vocab_size: raise ValueError("top_k must be between 1 and vocab_size")
        was_training=self.training; self.eval()
        try:
            for _ in range(max_new_tokens):
                context=idx[:,-self.config.block_size:]; logits,_=self(context); logits=logits[:,-1,:]/temperature
                if top_k is not None:
                    values,_=torch.topk(logits,top_k); logits[logits<values[:,-1,None]]=-float("inf")
                nxt=logits.argmax(-1,keepdim=True) if greedy else torch.multinomial(torch.softmax(logits,dim=-1),1)
                idx=torch.cat((idx,nxt),dim=1)
            return idx
        finally: self.train(was_training)
