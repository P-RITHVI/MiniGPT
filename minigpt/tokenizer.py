import json
import re
from collections import Counter
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


class BPETokenizer:
    """Small, deterministic byte-pair tokenizer that preserves the source text exactly."""
    def __init__(self, pieces, merges):
        self.itos = ["<unk>", *pieces]
        self.stoi = {piece: i for i, piece in enumerate(self.itos)}
        self.merges = [tuple(pair) for pair in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @property
    def vocab_size(self): return len(self.itos)

    @classmethod
    def train(cls, text, vocab_size=512):
        if not text: raise ValueError("Cannot fit tokenizer on empty training text")
        words = Counter(re.findall(r"\s+|[^\s]+", text))
        pieces = sorted(set("".join(words)))
        if vocab_size < len(pieces) + 1: raise ValueError("BPE vocab_size must fit all characters plus <unk>")
        sequences = Counter({tuple(word): count for word, count in words.items()})
        merges = []
        while len(pieces) + len(merges) + 1 < vocab_size:
            counts = {}
            for sequence, frequency in sequences.items():
                for pair in zip(sequence, sequence[1:]): counts[pair] = counts.get(pair, 0) + frequency
            if not counts: break
            candidates = [(count, pair) for pair, count in counts.items() if "".join(pair) not in pieces]
            if not candidates: break
            frequency, pair = max(candidates, key=lambda item: (item[0], item[1]))
            if frequency < 2: break
            merged = "".join(pair)
            merges.append(pair)
            pieces.append(merged)
            updated = Counter()
            for sequence, count in sequences.items():
                result = []; i = 0
                while i < len(sequence):
                    if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
                        result.append(merged); i += 2
                    else:
                        result.append(sequence[i]); i += 1
                updated[tuple(result)] += count
            sequences = updated
        return cls(pieces, merges)

    def encode(self, text):
        tokens = []
        for segment in re.findall(r"\s+|[^\s]+", text):
            sequence = list(segment)
            while len(sequence) > 1:
                candidates = [(self.ranks[pair], pair) for pair in zip(sequence, sequence[1:]) if pair in self.ranks]
                if not candidates: break
                _, pair = min(candidates)
                merged = "".join(pair); result = []; i = 0
                while i < len(sequence):
                    if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
                        result.append(merged); i += 2
                    else:
                        result.append(sequence[i]); i += 1
                sequence = result
            tokens.extend(sequence)
        return [self.stoi.get(token, 0) for token in tokens]

    def decode(self, ids):
        output = []
        for token_id in ids:
            if not isinstance(token_id, int) or token_id < 0 or token_id >= self.vocab_size: raise ValueError(f"Invalid token id: {token_id}")
            if token_id: output.append(self.itos[token_id])
        return "".join(output)

    def save(self, path):
        Path(path).write_text(json.dumps({"type": "bpe", "pieces": self.itos[1:], "merges": self.merges}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "bpe": raise ValueError("Malformed BPE tokenizer")
        return cls(data["pieces"], data["merges"])


def load_tokenizer(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return BPETokenizer.load(path) if isinstance(data, dict) and data.get("type") == "bpe" else CharTokenizer.load(path)
