#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from heuristic_secrets.models.scatter_attention import (
    HierarchicalLocalAttention,
    LocalAttentionND,
    RMSNorm,
    sinusoidal_pos_embed_nd,
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
    def __init__(self, width: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.dropout = dropout
        
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.out(out)


class AttentionBlock(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = SDPAttention(width, num_heads, dropout)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HierarchicalBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int = 17, n_levels: int = 4, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.hier_attn = HierarchicalLocalAttention(width, kernel_size, n_levels)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.hier_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int = 17, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.depthwise = nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x).transpose(1, 2)
        h = self.pointwise(self.depthwise(h)).transpose(1, 2)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SDPAttention2D(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.dropout = dropout
        
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        L = H * W
        
        x_flat = x.reshape(B, L, C)
        qkv = self.qkv(x_flat).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        
        out = out.transpose(1, 2).reshape(B, H, W, C)
        return self.out(out)


class AttentionBlock2D(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = SDPAttention2D(width, num_heads, dropout)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LocalBlock2D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 7, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.local_attn = LocalAttentionND(width, kernel_size, ndim=2)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock2D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 7, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.depthwise = nn.Conv2d(width, width, kernel_size, padding=kernel_size // 2, groups=width)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        h = self.norm1(x).permute(0, 3, 1, 2)
        h = self.pointwise(self.depthwise(h)).permute(0, 2, 3, 1)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SequenceClassifier(nn.Module):
    def __init__(self, block: nn.Module, width: int, n_layers: int, n_classes: int, seq_len: int):
        super().__init__()
        self.embed = nn.Linear(1, width)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, width) * 0.02)
        self.layers = nn.ModuleList([block for _ in range(n_layers)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x.unsqueeze(-1)) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


class ImageClassifier(nn.Module):
    def __init__(self, block: nn.Module, width: int, n_layers: int, n_classes: int, img_size: tuple[int, int]):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = nn.Linear(1, width)
        self.register_buffer(
            'pos_embed',
            sinusoidal_pos_embed_nd(img_size, width, torch.device('cpu'), torch.float32)
        )
        self.layers = nn.ModuleList([block for _ in range(n_layers)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, *self.img_size, 1)
        x = self.patch_embed(x) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).mean(dim=(1, 2))
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


def train_model(model, train_loader, test_loader, device, epochs, lr, verbose=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        if verbose:
            print(f'Epoch {epoch+1:2d}: train_acc={train_acc:.4f} test_acc={test_acc:.4f}')
    
    _, final_acc = evaluate(model, test_loader, device)
    return final_acc


def build_model(model_type, layers, n_classes, seq_len, device):
    WIDTH_ATTN = 64
    WIDTH_HIER = 53
    WIDTH_CONV = 70
    
    if model_type == 'attention':
        block_fn = lambda: AttentionBlock(WIDTH_ATTN, num_heads=4, mlp_mult=4)
        width = WIDTH_ATTN
    elif model_type == 'hier':
        block_fn = lambda: HierarchicalBlock(WIDTH_HIER, kernel_size=17, n_levels=4, mlp_mult=4)
        width = WIDTH_HIER
    elif model_type == 'conv':
        block_fn = lambda: ConvBlock(WIDTH_CONV, kernel_size=17, mlp_mult=4)
        width = WIDTH_CONV
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    model = SequenceClassifier(block_fn(), width, layers, n_classes, seq_len)
    model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
    return model.to(device)


def build_model_2d(model_type, layers, n_classes, img_size, device):
    WIDTH_ATTN = 48
    WIDTH_LOCAL = 48
    WIDTH_CONV = 52
    
    if model_type == 'attention':
        block_fn = lambda: AttentionBlock2D(WIDTH_ATTN, num_heads=4, mlp_mult=4)
        width = WIDTH_ATTN
    elif model_type == 'local':
        block_fn = lambda: LocalBlock2D(WIDTH_LOCAL, kernel_size=7, mlp_mult=4)
        width = WIDTH_LOCAL
    elif model_type == 'conv':
        block_fn = lambda: ConvBlock2D(WIDTH_CONV, kernel_size=7, mlp_mult=4)
        width = WIDTH_CONV
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    model = ImageClassifier(block_fn(), width, layers, n_classes, img_size)
    model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
    return model.to(device)


def load_dataset(name, batch_size):
    if name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('data', train=False, download=True, transform=transform)
        n_classes, seq_len, img_size = 10, 784, (28, 28)
        
    elif name == 'fashion':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_data = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
        n_classes, seq_len, img_size = 10, 784, (28, 28)
        
    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.4734,), (0.2516,))
        ])
        train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)
        n_classes, seq_len, img_size = 10, 1024, (32, 32)
        
    elif name == 'cifar100':
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.4734,), (0.2516,))
        ])
        train_data = datasets.CIFAR100('data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100('data', train=False, download=True, transform=transform)
        n_classes, seq_len, img_size = 100, 1024, (32, 32)
        
    else:
        raise ValueError(f'Unknown dataset: {name}')
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, n_classes, seq_len, img_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion', 'cifar10', 'cifar100'], default='mnist')
    parser.add_argument('--device', type=str, choices=['cpu', 'mps', 'cuda', 'auto'], default='auto')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model', type=str, default='all')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--2d', dest='mode_2d', action='store_true', help='Use 2D models (image-native)')
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    mode = '2D' if args.mode_2d else '1D'
    print(f'Device: {device}')
    print(f'Dataset: {args.dataset}')
    print(f'Mode: {mode}')
    
    train_loader, test_loader, n_classes, seq_len, img_size = load_dataset(args.dataset, args.batch_size)
    
    if args.mode_2d:
        all_model_types = ['attention', 'local', 'conv']
        builder = lambda mt: build_model_2d(mt, args.layers, n_classes, img_size, device)
        shape_str = f'img_size={img_size}'
    else:
        all_model_types = ['attention', 'hier', 'conv']
        builder = lambda mt: build_model(mt, args.layers, n_classes, seq_len, device)
        shape_str = f'seq_len={seq_len}'
    
    model_types = all_model_types if args.model == 'all' else [args.model]
    
    print(f'\nModel parameters ({args.layers} layers, {shape_str}, n_classes={n_classes}):')
    for mt in model_types:
        model = builder(mt)
        print(f'  {mt:12s}: {count_params(model):,} params')
    
    results = {mt: [] for mt in model_types}
    
    for run in range(args.runs):
        seed = run * 42
        torch.manual_seed(seed)
        
        if args.runs > 1:
            print(f'\n{"="*60}')
            print(f'Run {run+1}/{args.runs} (seed={seed})')
            print('='*60)
        
        for mt in model_types:
            torch.manual_seed(seed)
            model = builder(mt)
            
            print(f'\nTraining {mt}...')
            acc = train_model(model, train_loader, test_loader, device, args.epochs, args.lr, verbose=(args.runs == 1))
            results[mt].append(acc)
            print(f'{mt}: {acc:.4f}')
    
    print(f'\n{"="*60}')
    print('Final Results')
    print('='*60)
    
    for mt in model_types:
        accs = results[mt]
        mean_acc = sum(accs) / len(accs)
        if len(accs) > 1:
            std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
            print(f'{mt:12s}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(accs)})')
        else:
            print(f'{mt:12s}: {mean_acc:.4f}')


if __name__ == '__main__':
    main()
