import json
from pathlib import Path
import torch
from .config import Config
from .model import GPTLanguageModel
from .tokenizer import load_tokenizer

def save_checkpoint(path,model,tokenizer,config,extra=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tokenizer.save(path.with_suffix(".tokenizer.json")); config.save(path.with_suffix(".config.json"))
    torch.save({"model":model.state_dict(),"extra":extra or {}},path)

def load_checkpoint(path,device="cpu"):
    path=Path(path); config=Config.from_json(path.with_suffix(".config.json")); tokenizer=load_tokenizer(path.with_suffix(".tokenizer.json"))
    if config.vocab_size!=tokenizer.vocab_size: raise ValueError("Checkpoint and tokenizer vocabulary sizes differ")
    model=GPTLanguageModel(config).to(device)
    try: state=torch.load(path,map_location=device,weights_only=True)
    except TypeError: state=torch.load(path,map_location=device)
    model.load_state_dict(state["model"]); model.eval(); return model,tokenizer,config,state.get("extra",{})
