import math, random, time
import csv
from pathlib import Path
import torch
from .checkpoint import save_checkpoint
from .dataset import NextTokenDataset, random_batch, split_text
from .model import GPTLanguageModel
from .tokenizer import CharTokenizer

def evaluate(model,dataset,config):
    model.eval(); losses=[]
    with torch.no_grad():
        for _ in range(config.eval_batches):
            x,y=random_batch(dataset,config.batch_size,config.device); _,loss=model(x,y); losses.append(loss.item())
    model.train(); mean=sum(losses)/len(losses); return mean, math.exp(mean) if mean<700 else float("inf")

def train(config, data_path=None, steps=None):
    random.seed(config.seed); torch.manual_seed(config.seed)
    if config.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    text=Path(data_path or config.data_path).read_text(encoding="utf-8")
    train_text,val_text,_=split_text(text)
    tokenizer=CharTokenizer.train(train_text)
    train_data=NextTokenDataset(tokenizer.encode(train_text),config.block_size); val_data=NextTokenDataset(tokenizer.encode(val_text),config.block_size)
    if not len(train_data) or not len(val_data): raise ValueError("Train and validation splits must each exceed block_size")
    config.vocab_size=tokenizer.vocab_size; config.validate(); model=GPTLanguageModel(config).to(config.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    run_steps=steps if steps is not None else config.steps
    if run_steps < 1: raise ValueError("steps must be positive")
    history=[]; best_val=float("inf"); best_step=0; best_state=None
    start=time.perf_counter(); model.train()
    for step in range(1,run_steps+1):
        x,y=random_batch(train_data,config.batch_size,config.device); _,loss=model(x,y)
        if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip); optimizer.step()
        if step==1 or step%config.eval_interval==0 or step==run_steps:
            val_loss,ppl=evaluate(model,val_data,config)
            history.append({"step":step,"train_loss":loss.item(),"val_loss":val_loss,"val_perplexity":ppl,"elapsed_seconds":time.perf_counter()-start})
            if val_loss < best_val:
                best_val=val_loss; best_step=step
                best_state={name: tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()}
    output=Path(config.output_dir); output.mkdir(parents=True,exist_ok=True)
    model.load_state_dict(best_state)
    elapsed=time.perf_counter()-start
    extra={"history":history,"elapsed_seconds":elapsed,"best_step":best_step,"best_val_loss":best_val,
           "parameter_count":sum(p.numel() for p in model.parameters()),"training_steps":run_steps}
    save_checkpoint(output/"minigpt.pt",model,tokenizer,config,extra)
    with (output/"training_history.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    if len(history)>=3:
        try:
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(); ax.plot([r["step"] for r in history],[r["train_loss"] for r in history],label="training")
            ax.plot([r["step"] for r in history],[r["val_loss"] for r in history],label="validation")
            ax.set(xlabel="Step",ylabel="Cross-entropy loss",title="MiniGPT training history"); ax.legend(); fig.tight_layout()
            fig.savefig(output/"loss_curve.png",dpi=150); plt.close(fig)
        except ImportError:
            pass
    return model,tokenizer,history
