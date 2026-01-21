"""
Synthetic sequence tasks for benchmarking attention mechanisms.
Inspired by HazyResearch/zoology but standalone implementation.
"""

from dataclasses import dataclass
from typing import Literal
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


@dataclass
class SyntheticData:
    inputs: torch.Tensor
    labels: torch.Tensor
    task: str
    config: dict


class SyntheticDataset(Dataset):
    def __init__(self, data: SyntheticData):
        self.inputs = data.inputs
        self.labels = data.labels

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]


def mqar(
    vocab_size: int = 256,
    seq_len: int = 64,
    num_kv_pairs: int = 8,
    num_examples: int = 10000,
    power_a: float = 0.01,
    random_non_queries: bool = True,
    seed: int = 42,
) -> SyntheticData:
    """
    Multi-Query Associative Recall.
    
    Format: K1 V1 K2 V2 ... padding ... Q1 ... Q2 ...
    Labels: -100 for non-query positions, V_i at query Q_i position.
    
    Tests: in-context associative memory retrieval.
    """
    rng = np.random.default_rng(seed)
    
    key_vocab_size = vocab_size // 2
    
    all_inputs = []
    all_labels = []
    
    for _ in range(num_examples):
        keys = rng.choice(key_vocab_size, size=num_kv_pairs, replace=False)
        values = rng.integers(key_vocab_size, vocab_size, size=num_kv_pairs)
        
        kv_section = np.empty(num_kv_pairs * 2, dtype=np.int64)
        kv_section[0::2] = keys
        kv_section[1::2] = values
        
        remaining = seq_len - num_kv_pairs * 3
        
        if power_a > 0:
            p = np.arange(1, num_kv_pairs + 1, dtype=np.float64) ** (-power_a)
            p = p / p.sum()
        else:
            p = None
        
        query_indices = rng.choice(num_kv_pairs, size=num_kv_pairs, replace=True, p=p)
        queries = keys[query_indices]
        answers = values[query_indices]
        
        if random_non_queries:
            padding = rng.integers(0, key_vocab_size, size=remaining)
        else:
            padding = np.zeros(remaining, dtype=np.int64)
        
        inp = np.concatenate([kv_section, padding, queries])
        
        labels = np.full(seq_len, -100, dtype=np.int64)
        query_start = seq_len - num_kv_pairs
        labels[query_start:] = answers
        
        all_inputs.append(inp)
        all_labels.append(labels)
    
    return SyntheticData(
        inputs=torch.tensor(np.array(all_inputs), dtype=torch.long),
        labels=torch.tensor(np.array(all_labels), dtype=torch.long),
        task="mqar",
        config=dict(vocab_size=vocab_size, seq_len=seq_len, num_kv_pairs=num_kv_pairs),
    )


def compositional_mqar(
    vocab_size: int = 256,
    seq_len: int = 128,
    num_kv_pairs: int = 8,
    num_examples: int = 10000,
    seed: int = 42,
) -> SyntheticData:
    """
    Compositional MQAR: compound keys (K1, K2) -> V.
    Query gives both keys, model must retrieve value.
    
    Tests: compositional reasoning over multiple keys.
    """
    rng = np.random.default_rng(seed)
    
    key_vocab_size = vocab_size // 3
    
    all_inputs = []
    all_labels = []
    
    for _ in range(num_examples):
        keys1 = rng.choice(key_vocab_size, size=num_kv_pairs, replace=False)
        keys2 = rng.choice(key_vocab_size, size=num_kv_pairs, replace=False) + key_vocab_size
        values = rng.integers(2 * key_vocab_size, vocab_size, size=num_kv_pairs)
        
        kv_section = np.empty(num_kv_pairs * 3, dtype=np.int64)
        kv_section[0::3] = keys1
        kv_section[1::3] = keys2
        kv_section[2::3] = values
        
        num_queries = num_kv_pairs
        query_section_len = num_queries * 3
        remaining = seq_len - len(kv_section) - query_section_len
        
        padding = rng.integers(0, key_vocab_size, size=max(0, remaining))
        
        query_indices = rng.choice(num_kv_pairs, size=num_queries, replace=True)
        query_section = np.empty(query_section_len, dtype=np.int64)
        query_section[0::3] = keys1[query_indices]
        query_section[1::3] = keys2[query_indices]
        query_section[2::3] = 0
        
        inp = np.concatenate([kv_section, padding, query_section])[:seq_len]
        
        labels = np.full(seq_len, -100, dtype=np.int64)
        query_start = len(kv_section) + len(padding)
        for i in range(num_queries):
            answer_pos = query_start + i * 3 + 2
            if answer_pos < seq_len:
                labels[answer_pos] = values[query_indices[i]]
        
        all_inputs.append(inp)
        all_labels.append(labels)
    
    return SyntheticData(
        inputs=torch.tensor(np.array(all_inputs), dtype=torch.long),
        labels=torch.tensor(np.array(all_labels), dtype=torch.long),
        task="compositional_mqar",
        config=dict(vocab_size=vocab_size, seq_len=seq_len, num_kv_pairs=num_kv_pairs),
    )


def forgetting_mqar(
    vocab_size: int = 256,
    seq_len: int = 128,
    num_kv_pairs: int = 8,
    num_updates: int = 2,
    num_examples: int = 10000,
    seed: int = 42,
) -> SyntheticData:
    """
    Forgetting MQAR: keys reappear with new values.
    Model must remember the LATEST value for each key.
    
    Tests: memory update / forgetting capability.
    """
    rng = np.random.default_rng(seed)
    
    key_vocab_size = vocab_size // 2
    
    all_inputs = []
    all_labels = []
    
    for _ in range(num_examples):
        keys = rng.choice(key_vocab_size, size=num_kv_pairs, replace=False)
        
        kv_sections = []
        latest_values = {}
        
        for update_idx in range(num_updates):
            values = rng.integers(key_vocab_size, vocab_size, size=num_kv_pairs)
            section = np.empty(num_kv_pairs * 2, dtype=np.int64)
            section[0::2] = keys
            section[1::2] = values
            kv_sections.append(section)
            
            for k, v in zip(keys, values):
                latest_values[k] = v
        
        all_kv = np.concatenate(kv_sections)
        
        remaining = seq_len - len(all_kv) - num_kv_pairs
        padding = rng.integers(0, key_vocab_size, size=max(0, remaining))
        
        query_indices = rng.choice(num_kv_pairs, size=num_kv_pairs, replace=False)
        queries = keys[query_indices]
        
        inp = np.concatenate([all_kv, padding, queries])[:seq_len]
        
        labels = np.full(seq_len, -100, dtype=np.int64)
        query_start = len(all_kv) + len(padding)
        for i, q in enumerate(queries):
            pos = query_start + i
            if pos < seq_len:
                labels[pos] = latest_values[q]
        
        all_inputs.append(inp)
        all_labels.append(labels)
    
    return SyntheticData(
        inputs=torch.tensor(np.array(all_inputs), dtype=torch.long),
        labels=torch.tensor(np.array(all_labels), dtype=torch.long),
        task="forgetting_mqar",
        config=dict(vocab_size=vocab_size, seq_len=seq_len, num_kv_pairs=num_kv_pairs, num_updates=num_updates),
    )


def parity(
    seq_len: int = 64,
    num_examples: int = 10000,
    seed: int = 42,
) -> SyntheticData:
    """
    Parity task: output XOR of all input bits.
    
    Input: sequence of 0s and 1s
    Label: single parity bit at the end (-100 elsewhere)
    
    Tests: global information aggregation.
    """
    rng = np.random.default_rng(seed)
    
    inputs = rng.integers(0, 2, size=(num_examples, seq_len))
    parities = inputs.sum(axis=1) % 2
    
    labels = np.full((num_examples, seq_len), -100, dtype=np.int64)
    labels[:, -1] = parities
    
    return SyntheticData(
        inputs=torch.tensor(inputs, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        task="parity",
        config=dict(vocab_size=2, seq_len=seq_len),
    )


def cumulative_parity(
    seq_len: int = 64,
    num_examples: int = 10000,
    seed: int = 42,
) -> SyntheticData:
    """
    Cumulative parity: running XOR at each position.
    
    Input: 0 1 1 0 1
    Label: 0 1 0 0 1
    
    Tests: sequential state tracking.
    """
    rng = np.random.default_rng(seed)
    
    inputs = rng.integers(0, 2, size=(num_examples, seq_len))
    labels = np.cumsum(inputs, axis=1) % 2
    
    return SyntheticData(
        inputs=torch.tensor(inputs, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        task="cumulative_parity",
        config=dict(vocab_size=2, seq_len=seq_len),
    )


def majority(
    seq_len: int = 64,
    num_examples: int = 10000,
    seed: int = 42,
) -> SyntheticData:
    """
    Majority task: output 1 if more 1s than 0s.
    
    Tests: counting / aggregation.
    """
    rng = np.random.default_rng(seed)
    
    inputs = rng.integers(0, 2, size=(num_examples, seq_len))
    majorities = (inputs.sum(axis=1) > seq_len // 2).astype(np.int64)
    
    labels = np.full((num_examples, seq_len), -100, dtype=np.int64)
    labels[:, -1] = majorities
    
    return SyntheticData(
        inputs=torch.tensor(inputs, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        task="majority",
        config=dict(vocab_size=2, seq_len=seq_len),
    )


def cumulative_majority(
    seq_len: int = 64,
    num_examples: int = 10000,
    seed: int = 42,
) -> SyntheticData:
    """
    Cumulative majority: running majority at each position.
    
    Tests: running count state tracking.
    """
    rng = np.random.default_rng(seed)
    
    inputs = rng.integers(0, 2, size=(num_examples, seq_len))
    cumsum = np.cumsum(inputs, axis=1)
    positions = np.arange(1, seq_len + 1)
    labels = (cumsum > positions // 2).astype(np.int64)
    
    return SyntheticData(
        inputs=torch.tensor(inputs, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        task="cumulative_majority",
        config=dict(vocab_size=2, seq_len=seq_len),
    )


def copying(
    vocab_size: int = 16,
    seq_len: int = 128,
    copy_len: int = 16,
    num_examples: int = 10000,
    seed: int = 42,
) -> SyntheticData:
    """
    Copying task: memorize prefix, reproduce after delay.
    
    Format: [tokens to copy] [zeros...] [delimiter] [reproduce here]
    
    Tests: long-range memory.
    """
    rng = np.random.default_rng(seed)
    
    delimiter = vocab_size
    effective_vocab = vocab_size + 1
    
    all_inputs = []
    all_labels = []
    
    delay_len = seq_len - 2 * copy_len - 1
    
    for _ in range(num_examples):
        to_copy = rng.integers(0, vocab_size, size=copy_len)
        delay = np.zeros(delay_len, dtype=np.int64)
        
        inp = np.concatenate([to_copy, delay, [delimiter], np.zeros(copy_len, dtype=np.int64)])
        
        labels = np.full(seq_len, -100, dtype=np.int64)
        labels[-copy_len:] = to_copy
        
        all_inputs.append(inp)
        all_labels.append(labels)
    
    return SyntheticData(
        inputs=torch.tensor(np.array(all_inputs), dtype=torch.long),
        labels=torch.tensor(np.array(all_labels), dtype=torch.long),
        task="copying",
        config=dict(vocab_size=effective_vocab, seq_len=seq_len, copy_len=copy_len),
    )


def selective_copying(
    vocab_size: int = 16,
    seq_len: int = 128,
    num_tokens: int = 16,
    num_examples: int = 10000,
    seed: int = 42,
) -> SyntheticData:
    """
    Selective copying: copy only marked tokens.
    
    Tokens are interspersed with noise. Marked tokens (MSB=1) should be copied.
    
    Tests: selective attention / filtering.
    """
    rng = np.random.default_rng(seed)
    
    marker_bit = vocab_size
    effective_vocab = vocab_size * 2
    
    all_inputs = []
    all_labels = []
    
    content_len = seq_len - num_tokens - 1
    
    for _ in range(num_examples):
        positions = np.sort(rng.choice(content_len, size=num_tokens, replace=False))
        tokens = rng.integers(0, vocab_size, size=num_tokens)
        
        content = rng.integers(0, vocab_size, size=content_len)
        content[positions] = tokens + marker_bit
        
        delimiter = np.array([effective_vocab - 1], dtype=np.int64)
        query = np.zeros(num_tokens, dtype=np.int64)
        
        inp = np.concatenate([content, delimiter, query])
        
        labels = np.full(seq_len, -100, dtype=np.int64)
        labels[-num_tokens:] = tokens
        
        all_inputs.append(inp)
        all_labels.append(labels)
    
    return SyntheticData(
        inputs=torch.tensor(np.array(all_inputs), dtype=torch.long),
        labels=torch.tensor(np.array(all_labels), dtype=torch.long),
        task="selective_copying",
        config=dict(vocab_size=effective_vocab, seq_len=seq_len, num_tokens=num_tokens),
    )


def induction_heads(
    vocab_size: int = 64,
    seq_len: int = 256,
    pattern_len: int = 2,
    num_examples: int = 10000,
    seed: int = 42,
) -> SyntheticData:
    """
    Induction heads task: [A][B]...[A] -> predict [B].
    
    Tests: pattern matching / in-context learning.
    """
    rng = np.random.default_rng(seed)
    
    all_inputs = []
    all_labels = []
    
    for _ in range(num_examples):
        pattern = rng.integers(0, vocab_size, size=pattern_len)
        
        first_pos = rng.integers(0, seq_len // 4)
        second_pos = rng.integers(seq_len // 2, seq_len - pattern_len)
        
        inp = rng.integers(0, vocab_size, size=seq_len)
        inp[first_pos:first_pos + pattern_len] = pattern
        inp[second_pos:second_pos + pattern_len - 1] = pattern[:-1]
        inp[second_pos + pattern_len - 1] = 0
        
        labels = np.full(seq_len, -100, dtype=np.int64)
        labels[second_pos + pattern_len - 1] = pattern[-1]
        
        all_inputs.append(inp)
        all_labels.append(labels)
    
    return SyntheticData(
        inputs=torch.tensor(np.array(all_inputs), dtype=torch.long),
        labels=torch.tensor(np.array(all_labels), dtype=torch.long),
        task="induction_heads",
        config=dict(vocab_size=vocab_size, seq_len=seq_len, pattern_len=pattern_len),
    )


TASKS = {
    "mqar": mqar,
    "compositional_mqar": compositional_mqar,
    "forgetting_mqar": forgetting_mqar,
    "parity": parity,
    "cumulative_parity": cumulative_parity,
    "majority": majority,
    "cumulative_majority": cumulative_majority,
    "copying": copying,
    "selective_copying": selective_copying,
    "induction_heads": induction_heads,
}


def load_task(
    task: str,
    seq_len: int = 128,
    num_train: int = 50000,
    num_test: int = 10000,
    batch_size: int = 64,
    seed: int = 42,
    **kwargs,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Load a synthetic task with train/test dataloaders.
    
    Returns: (train_loader, test_loader, config_dict)
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASKS.keys())}")
    
    task_fn = TASKS[task]
    
    common_kwargs = dict(seq_len=seq_len, **kwargs)
    if "vocab_size" not in kwargs and task not in ("parity", "cumulative_parity", "majority", "cumulative_majority"):
        common_kwargs["vocab_size"] = 256
    
    train_data = task_fn(num_examples=num_train, seed=seed, **common_kwargs)
    test_data = task_fn(num_examples=num_test, seed=seed + 1000, **common_kwargs)
    
    train_loader = DataLoader(
        SyntheticDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        SyntheticDataset(test_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, test_loader, train_data.config
