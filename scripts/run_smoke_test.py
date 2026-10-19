from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.config import Config
from minigpt.train import train

def main():
    c=Config(block_size=32,d_model=32,n_heads=4,n_layers=1,batch_size=8,steps=5,eval_interval=5,eval_batches=2,output_dir="artifacts/smoke")
    _,_,history=train(c,steps=5); print("Smoke training completed:",history[-1])
if __name__=="__main__": main()
