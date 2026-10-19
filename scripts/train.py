import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.config import Config
from minigpt.train import train

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.json"); p.add_argument("--steps",type=int); p.add_argument("--data",help="Override the dataset path from the config"); p.add_argument("--output-dir",help="Override the checkpoint and metrics directory")
    a=p.parse_args(); c=Config.from_json(a.config)
    if a.output_dir: c.output_dir=a.output_dir
    _,_,history=train(c,data_path=a.data,steps=a.steps); print(history[-1])
if __name__=="__main__": main()
