from pathlib import Path
from minigpt.config import Config
from minigpt.train import train

if __name__=="__main__":
    config=Config.from_json("configs/default.json"); config.steps=20; config.output_dir="artifacts/quickstart"
    train(config); print("Checkpoint saved under",Path(config.output_dir).resolve())
