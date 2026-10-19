from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Config:
    vocab_size: int = 0
    block_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 16
    steps: int = 300
    eval_interval: int = 50
    eval_batches: int = 10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "cpu"
    data_path: str = "data/tiny_corpus.txt"
    output_dir: str = "artifacts"
    @property
    def ff_dim(self): return 4 * self.d_model
    def validate(self):
        if self.vocab_size < 0 or self.block_size < 1 or self.d_model < 1 or self.n_heads < 1 or self.n_layers < 1:
            raise ValueError("Model dimensions must be positive and vocab_size nonnegative")
        if self.d_model % self.n_heads: raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1: raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.batch_size < 1 or self.steps < 1: raise ValueError("Training parameters must be positive")
    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(**data); config.validate(); return config
    def save(self, path): Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
