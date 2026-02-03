# Linear Attention MLP Integration Guide

## Overview

Replace standard MLP (`relu(x @ W1)² @ W2`) with linear attention where ReLU² serves as both the activation and the attention kernel.

## Changes Required

### 1. Add to imports (top of train_gpt.py)

```python
# At line ~37, add:
from linear_attn_mlp import LinearAttnMLP, linear_attn_relu2_ref, linear_attn_relu2_full
```

### 2. Add Hyperparameter flag

```python
# In Hyperparameters dataclass (~line 1493), add:
@dataclass
class Hyperparameters:
    ...
    use_diff_attn: bool = False
    use_linear_attn_mlp: bool = False  # NEW: Use linear attention in MLP
    linear_attn_mlp_full: bool = False  # NEW: Use full outer-product attention (slower but more expressive)
```

### 3. Modify MLP class

Replace the current MLP class (~line 1033):

```python
class MLP(nn.Module):
    def __init__(self, use_linear_attn: bool = False, use_full_attn: bool = False, 
                 dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.use_linear_attn = use_linear_attn
        
        if use_linear_attn:
            # Linear attention MLP parameters
            self.dim = dim
            self.hidden_dim = 2 * dim  # 1536 - matches param count
            self.num_heads = num_heads
            self.head_dim = self.hidden_dim // num_heads  # 128
            self.use_full_attn = use_full_attn
            self.eps = 1e-6
        # Standard MLP: weights passed via forward()

    def forward(self, x: Tensor, c_fc: Tensor = None, c_proj: Tensor = None,
                w_qkv: Tensor = None, w_out: Tensor = None):
        if self.use_linear_attn:
            return self._forward_linear_attn(x, w_qkv, w_out)
        else:
            # Original: relu(x)^2 fused kernel
            return FusedLinearReLUSquareFunction.apply(x, c_fc, c_proj)
    
    def _forward_linear_attn(self, x: Tensor, w_qkv: Tensor, w_out: Tensor) -> Tensor:
        """Linear attention MLP forward pass."""
        B, T, _ = x.shape
        
        # Project to Q, K, V
        qkv = F.linear(x, w_qkv)  # (B, T, 3 * hidden_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to heads
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        
        # Linear attention with ReLU² kernel
        if self.use_full_attn:
            o = linear_attn_relu2_full(q, k, v, eps=self.eps)
        else:
            o = linear_attn_relu2_ref(q, k, v, eps=self.eps)
        
        # Reshape and project output
        o = o.view(B, T, self.hidden_dim)
        return F.linear(o, w_out)
```

### 4. Modify Block class

Update Block to pass through the linear attention flag (~line 1044):

```python
class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, has_attn: bool, has_mlp: bool, 
                 use_paired_head: bool, use_diff_attn: bool = False, layer_idx: int = 0,
                 use_linear_attn_mlp: bool = False, linear_attn_mlp_full: bool = False):
        super().__init__()
        self.attn = CausalSelfAttention(...) if has_attn else None
        self.mlp = MLP(
            use_linear_attn=use_linear_attn_mlp,
            use_full_attn=linear_attn_mlp_full,
            dim=dim,
            num_heads=num_heads
        ) if has_mlp else None
        self.use_linear_attn_mlp = use_linear_attn_mlp
```

### 5. Add new parameter bank for linear attention MLP weights

In GPT.__init__ (~line 1128), add a new bank:

```python
# Standard MLP bank (keep for non-linear-attn layers or backward compat)
self.mlp_bank = nn.Parameter(torch.empty(num_mlp_with_padding, 2, mlp_hdim, model_dim))

# NEW: Linear attention MLP bank
if use_linear_attn_mlp:
    linear_attn_hidden = 2 * model_dim  # 1536
    # Shape: (num_mlp_layers, 2, weight_dim, model_dim)
    # [0] = w_qkv: (3 * hidden, dim) = (4608, 768)
    # [1] = w_out: (dim, hidden) stored as (hidden, dim) = (1536, 768)
    self.linear_attn_mlp_bank = nn.Parameter(
        torch.empty(num_mlp_with_padding, 2, 3 * linear_attn_hidden, model_dim)
    )
    # Note: w_out is smaller, so we pad or use separate storage
    # Alternative: store separately
    self.linear_attn_qkv_bank = nn.Parameter(
        torch.empty(num_mlp_with_padding, 3 * linear_attn_hidden, model_dim)  # (12, 4608, 768)
    )
    self.linear_attn_out_bank = nn.Parameter(
        torch.empty(num_mlp_with_padding, model_dim, linear_attn_hidden)  # (12, 768, 1536)
    )
```

### 6. Update forward pass to use correct weights

In GPT.forward, when calling blocks:

```python
for i, block in enumerate(self.blocks):
    if block.mlp is not None:
        mlp_idx = self.layer_to_mlp_idx[i]
        if self.use_linear_attn_mlp:
            w_qkv = self.linear_attn_qkv_bank[mlp_idx]
            w_out = self.linear_attn_out_bank[mlp_idx]
            x = block(x, attn_args, qkvo_w=..., w_qkv=w_qkv, w_out=w_out)
        else:
            c_fc = self.mlp_bank[mlp_idx, 0]
            c_proj = self.mlp_bank[mlp_idx, 1]
            x = block(x, attn_args, qkvo_w=..., c_fc=c_fc, c_proj=c_proj)
```

## Parameter Count Comparison

| Component | Standard MLP | Linear Attn MLP |
|-----------|--------------|-----------------|
| W_up / W_qkv | (3072, 768) = 2.36M | (4608, 768) = 3.54M |
| W_down / W_out | (768, 3072) = 2.36M | (768, 1536) = 1.18M |
| **Total per layer** | **4.72M** | **4.72M** |

With `hidden_dim = 2 * dim`, parameter count matches exactly!

## Expected Behavior Changes

1. **Token Mixing**: Tokens can now influence each other through the MLP hidden layer
2. **Causal**: The linear attention is causal - position t only sees positions 0..t
3. **ReLU² Double Duty**: Serves as both nonlinearity and attention kernel
4. **Compute**: Similar FLOPs but different memory access pattern

## Optimization Path

1. **Phase 1**: Use reference PyTorch implementation (current)
2. **Phase 2**: Integrate fla's optimized chunk_linear_attn kernel
3. **Phase 3**: Write custom fused Triton kernel combining QKV projection + attention

## Integration with fla library

```python
# Install: pip install flash-linear-attention
from fla.ops.linear_attn import chunk_linear_attn

def forward_with_fla(self, x, w_qkv, w_out):
    qkv = F.linear(x, w_qkv)
    q, k, v = qkv.chunk(3, dim=-1)
    
    # Reshape to (B, T, H, D)
    q = q.view(B, T, self.num_heads, self.head_dim)
    k = k.view(B, T, self.num_heads, self.head_dim)
    v = v.view(B, T, self.num_heads, self.head_dim)
    
    # Apply ReLU² feature map
    q = F.relu(q) ** 2
    k = F.relu(k) ** 2
    
    # Use fla's optimized kernel
    o, _ = chunk_linear_attn(q, k, v, normalize=True, scale=1.0)
    
    o = o.view(B, T, -1)
    return F.linear(o, w_out)
```

## Quick Test

To test without full integration:

```python
# In train_gpt.py, temporarily replace MLP.forward:
def forward(self, x, c_fc, c_proj):
    B, T, D = x.shape
    
    # Simulate linear attention MLP with same weights
    # This is just for testing - uses c_fc/c_proj in a different way
    h = F.linear(x, c_fc[:D*3, :])  # Use subset as QKV
    q, k, v = h.view(B, T, 3, -1).unbind(2)
    
    q = F.relu(q) ** 2
    k = F.relu(k) ** 2
    
    # Simple element-wise linear attention
    kv_cumsum = (k * v).cumsum(dim=1)
    k_cumsum = k.cumsum(dim=1)
    o = q * kv_cumsum / (q * k_cumsum + 1e-6).sum(-1, keepdim=True)
    
    return F.linear(o, c_proj[:, :o.shape[-1]])
```
