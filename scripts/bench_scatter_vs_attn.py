#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from heuristic_secrets.models.scatter_attention import (
    InceptionScatterAttention,
    RMSNorm,
    apply_rope,
    sinusoidal_pos_embed,
)


class SwiGLU(nn.Module):
    def __init__(self, width: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = width * mult
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class SDPAttention(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.rope_base = rope_base
        self.dropout = dropout
        
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        
        x_pos = x + sinusoidal_pos_embed(L, C, x.device, x.dtype, self.rope_base)
        x_rope = apply_rope(x_pos, self.rope_base)
        
        qkv = self.qkv(x_rope).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.out(out)


class AttentionBlock(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, mlp_mult: int = 4, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = SDPAttention(width, num_heads, dropout, rope_base)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ScatterBlock(nn.Module):
    def __init__(self, width: int, scatter_channels: int = 4, num_samples: int = 16, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.scatter = InceptionScatterAttention(
            embed_dim=width,
            scatter_channels=scatter_channels,
            num_samples=num_samples,
        )
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scatter(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SequenceClassifier(nn.Module):
    def __init__(self, block: nn.Module, width: int, n_layers: int, n_classes: int, seq_len: int):
        super().__init__()
        self.embed = nn.Linear(1, width)
        self.layers = nn.ModuleList([block for _ in range(n_layers)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x.unsqueeze(-1))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, loader, optimizer, device, desc="Train"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")
    
    return total_loss / total, correct / total


def train_model(name, model, train_loader, test_loader, device, epochs, lr):
    print(f'\n{"="*60}')
    print(f'Training {name} model...')
    print('='*60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f'Epoch {epoch+1:2d}: train_acc={train_acc:.4f} test_acc={test_acc:.4f}')
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--device', type=str, choices=['cpu', 'mps', 'cuda', 'auto'], default='auto')
    parser.add_argument('--scatter-first', action='store_true', help='Train scatter model first')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')
    
    WIDTH = 64
    N_CLASSES = 10
    SEQ_LEN = 784
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
    
    attn_block = lambda: AttentionBlock(WIDTH, num_heads=4, mlp_mult=4)
    scatter_block = lambda: ScatterBlock(WIDTH, scatter_channels=4, num_samples=16, mlp_mult=4)
    
    attn_model = SequenceClassifier(attn_block(), WIDTH, args.layers, N_CLASSES, SEQ_LEN)
    scatter_model = SequenceClassifier(scatter_block(), WIDTH, args.layers, N_CLASSES, SEQ_LEN)
    
    attn_model.layers = nn.ModuleList([attn_block() for _ in range(args.layers)])
    scatter_model.layers = nn.ModuleList([scatter_block() for _ in range(args.layers)])
    attn_model = attn_model.to(device)
    scatter_model = scatter_model.to(device)
    
    print(f'\nAttention model:  {count_params(attn_model):,} params')
    print(f'Scatter model:    {count_params(scatter_model):,} params')
    print(f'Param ratio:      {count_params(scatter_model) / count_params(attn_model):.2f}x')
    
    models = [
        ('Scatter', scatter_model),
        ('Attention', attn_model),
    ] if args.scatter_first else [
        ('Attention', attn_model),
        ('Scatter', scatter_model),
    ]
    
    for name, model in models:
        train_model(name, model, train_loader, test_loader, device, args.epochs, args.lr)
    
    print(f'\n{"="*60}')
    print('Final Results')
    print('='*60)
    
    _, attn_final = evaluate(attn_model, test_loader, device)
    _, scatter_final = evaluate(scatter_model, test_loader, device)
    
    print(f'Attention:  {attn_final:.4f} ({count_params(attn_model):,} params)')
    print(f'Scatter:    {scatter_final:.4f} ({count_params(scatter_model):,} params)')


if __name__ == '__main__':
    main()
