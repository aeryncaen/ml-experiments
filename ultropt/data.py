"""
Shakespeare dataset: download, tokenize (char-level), and batch.
"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "shakespeare.txt")


def download_shakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print(f"Downloading Shakespeare to {DATA_FILE} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    with open(DATA_FILE, "r") as f:
        text = f.read()
    return text


class CharTokenizer:
    """Dead-simple char-level tokenizer."""

    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class ShakespeareDataset(Dataset):
    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def get_datasets(block_size=128, val_frac=0.1):
    """Return (train_dataset, val_dataset, tokenizer)."""
    text = download_shakespeare()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = len(data)
    split = int(n * (1 - val_frac))
    train_data = data[:split]
    val_data = data[split:]

    train_ds = ShakespeareDataset(train_data, block_size)
    val_ds = ShakespeareDataset(val_data, block_size)
    return train_ds, val_ds, tokenizer


def make_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
