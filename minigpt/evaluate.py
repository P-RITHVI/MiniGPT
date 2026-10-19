import math
import torch
from .dataset import NextTokenDataset
from .train import evaluate

def evaluate_split(model,tokenizer,text,config):
    data=NextTokenDataset(tokenizer.encode(text),config.block_size)
    if not len(data): raise ValueError("Evaluation split too short")
    model.eval(); losses=[]
    with torch.no_grad():
        for start in range(0,len(data),config.batch_size):
            batch=[data[i] for i in range(start,min(start+config.batch_size,len(data)))]
            x=torch.stack([p[0] for p in batch]).to(config.device); y=torch.stack([p[1] for p in batch]).to(config.device)
            _,loss=model(x,y); losses.append((loss.item(),len(batch)))
    mean=sum(l*n for l,n in losses)/sum(n for _,n in losses)
    return {"loss":mean,"perplexity":math.exp(mean) if mean<700 else None}
