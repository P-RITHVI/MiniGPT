import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minigpt.checkpoint import load_checkpoint
from minigpt.dataset import split_text
from minigpt.evaluate import evaluate_split
from minigpt.baseline import BigramBaseline

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--data",default=None); a=p.parse_args()
    model,tok,c,extra=load_checkpoint(a.checkpoint,cpu_device()); text=Path(a.data or c.data_path).read_text(encoding="utf-8"); _,_,test=split_text(text)
    train_text,_,test=split_text(text)
    test_metrics=evaluate_split(model,tok,test,c)
    baseline=BigramBaseline().fit([tok.encode(train_text)],tok.vocab_size)
    import math, json
    baseline_loss=baseline.loss([tok.encode(test)])
    result={"model":"MiniGPT","test":test_metrics,"baseline":{"name":"smoothed token bigram","test_loss":baseline_loss,"perplexity":math.exp(baseline_loss)},"training_history":extra.get("history",[])}
    output=Path(a.checkpoint).with_name("evaluation.json"); output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(result)
def cpu_device(): return "cpu"
if __name__=="__main__": main()
