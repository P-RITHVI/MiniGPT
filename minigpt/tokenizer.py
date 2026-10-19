import json
from pathlib import Path


class CharTokenizer:
    """Deterministic character tokenizer with a dedicated unknown token."""
    def __init__(self, chars):
        ordered = sorted(set(chars))
        self.itos = ["<unk>", *ordered]
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
    @classmethod
    def train(cls, text):
        if not text: raise ValueError("Cannot fit tokenizer on empty training text")
        return cls(text)
    @property
    def vocab_size(self): return len(self.itos)
    def encode(self, text): return [self.stoi.get(ch, 0) for ch in text]
    def decode(self, ids):
        output = []
        for token_id in ids:
            if not isinstance(token_id, int) or token_id < 0 or token_id >= self.vocab_size: raise ValueError(f"Invalid token id: {token_id}")
            if token_id: output.append(self.itos[token_id])
        return "".join(output)
    def save(self, path): Path(path).write_text(json.dumps(self.itos, ensure_ascii=False, indent=2), encoding="utf-8")
    @classmethod
    def load(cls, path):
        tokens = json.loads(Path(path).read_text(encoding="utf-8"))
        if not tokens or tokens[0] != "<unk>" or len(set(tokens)) != len(tokens): raise ValueError("Malformed tokenizer vocabulary")
        obj = cls.__new__(cls); obj.itos = tokens; obj.stoi = {s:i for i,s in enumerate(tokens)}; return obj
