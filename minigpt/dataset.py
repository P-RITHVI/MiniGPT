import torch
from torch.utils.data import Dataset


def split_text(text, train_fraction=.8, val_fraction=.1):
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1: raise ValueError("Fractions must be positive and sum to less than one")
    a, b = int(len(text)*train_fraction), int(len(text)*(train_fraction+val_fraction))
    return text[:a], text[a:b], text[b:]

class NextTokenDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = torch.as_tensor(tokens, dtype=torch.long); self.block_size = block_size
        if block_size < 1: raise ValueError("block_size must be positive")
    def __len__(self): return max(0, len(self.tokens)-self.block_size)
    def __getitem__(self, i):
        x = self.tokens[i:i+self.block_size]; return x, self.tokens[i+1:i+self.block_size+1]

def random_batch(dataset, batch_size, device="cpu", generator=None):
    if len(dataset) == 0: raise ValueError("Split is too short for one context window")
    idx = torch.randint(len(dataset), (batch_size,), generator=generator)
    x, y = zip(*(dataset[int(i)] for i in idx))
    return torch.stack(x).to(device), torch.stack(y).to(device)
