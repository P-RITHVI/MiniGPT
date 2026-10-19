import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from minigpt.checkpoint import load_checkpoint

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--prompt",required=True); p.add_argument("--tokens",type=int,default=200); p.add_argument("--temperature",type=float,default=1.0); p.add_argument("--top-k",type=int); p.add_argument("--greedy",action="store_true")
    a=p.parse_args(); model,tok,c,_=load_checkpoint(a.checkpoint,cpu_device()); ids=torch.tensor([tok.encode(a.prompt)],dtype=torch.long,device=c.device)
    print(tok.decode(model.generate(ids,a.tokens,a.temperature,a.top_k,a.greedy)[0].tolist()))
def cpu_device(): return "cpu"
if __name__=="__main__": main()
