# Fast Diffusion Methods for Sequence Mask Prediction

## Overview

This document covers fast diffusion techniques applicable to our use case: predicting binary masks over text sequences to identify secret spans.

## Key Techniques for Fast Diffusion

### 1. Consistency Models (Song et al., 2023)

**Core idea**: Train a model to map any point on the diffusion trajectory directly to the clean data, enabling single-step generation.

```
# Standard diffusion: x_T -> x_{T-1} -> ... -> x_0 (many steps)
# Consistency model: x_t -> x_0 (single step, any t)
```

**Why it matters for us**: 
- 1-2 step inference possible
- Can be distilled from pretrained diffusion models or trained from scratch
- Perfect for real-time mask prediction

**Implementation**: 
- `consistency_models` package (OpenAI)
- Key hyperparameter: skip connections in the denoising network

### 2. Rectified Flow / Flow Matching (Lipman et al., 2023; Liu et al., 2023)

**Core idea**: Learn straight-line paths between noise and data instead of curved diffusion trajectories.

```python
# Rectified flow training objective
def rectified_flow_loss(model, x_0, x_1):
    t = torch.rand(x_0.shape[0], 1)
    x_t = t * x_1 + (1 - t) * x_0  # Linear interpolation
    v_pred = model(x_t, t)
    v_target = x_1 - x_0  # Velocity is constant along straight path
    return F.mse_loss(v_pred, v_target)
```

**Advantages**:
- Simpler training than score matching
- Naturally few-step (paths are straight)
- State-of-art on many benchmarks (Stable Diffusion 3 uses this)

### 3. Latent Consistency Models (Luo et al., 2023)

**Core idea**: Apply consistency distillation in latent space.

- 1-4 step generation
- Used in SDXL-Turbo, LCM-LoRA
- Can fine-tune existing models for speed

### 4. Progressive Distillation (Salimans & Ho, 2022)

**Core idea**: Iteratively halve the number of steps by training student to match 2 teacher steps in 1.

```
Teacher: 1024 steps -> Student: 512 steps -> Student: 256 -> ... -> 4 steps
```

## Diffusion for Discrete/Sequential Data

### Challenges with Text/Masks

1. **Discrete tokens**: Standard diffusion assumes continuous space
2. **Variable length**: Sequences aren't fixed-size images
3. **Structure**: Masks have contiguous spans, not independent pixels

### Approaches

#### A. Continuous Relaxation (Our Recommended Approach)

For binary mask prediction, treat mask values as continuous in [0,1]:

```python
class MaskDiffusion(nn.Module):
    def __init__(self, encoder, denoiser):
        self.encoder = encoder  # Encodes input text
        self.denoiser = denoiser  # Predicts clean mask from noisy
    
    def forward(self, text, noisy_mask, t):
        text_features = self.encoder(text)
        # Denoiser sees: noisy mask + text conditioning + timestep
        clean_mask_pred = self.denoiser(noisy_mask, text_features, t)
        return clean_mask_pred
    
    def sample(self, text, steps=4):
        mask = torch.randn(text.shape[0], text.shape[1])  # Start from noise
        for t in reversed(range(steps)):
            mask = self.denoise_step(mask, text, t)
        return (mask > 0.5).float()  # Threshold to binary
```

**Why this works for us**:
- Mask is naturally continuous-ish (confidence per position)
- Same length as input (no length prediction needed)
- Final threshold gives discrete output

#### B. Discrete Diffusion (D3PM, Austin et al., 2021)

For truly discrete outputs, use transition matrices:

```python
# Instead of Gaussian noise, use discrete corruption
# States: {0, 1} for binary mask
# Transition: gradually flip bits toward uniform
```

Less relevant for us since masks are naturally soft.

#### C. Score Entropy Discrete Diffusion (Lou et al., 2024)

Recent work on discrete diffusion with better training objectives. Worth watching but more complex.

### Relevant Papers for Sequence Masking

1. **DiffuMask** (Wu et al., 2023) - Diffusion for image segmentation, directly applicable to 1D
2. **SegDiff** (Amit et al., 2023) - Diffusion segmentation, iterative refinement of masks
3. **Bit Diffusion** (Chen et al., 2023) - Diffusion over binary data

## Architecture Recommendations

### For Mask Prediction Specifically

```python
class MaskDiffusionModel(nn.Module):
    """
    Input: text sequence [B, L]
    Output: mask [B, L] in [0, 1]
    """
    def __init__(self, vocab_size=256, d_model=256, n_layers=4):
        # Text encoder (frozen or learned)
        self.text_encoder = ByteEncoder(vocab_size, d_model)
        
        # Denoiser: takes noisy mask + text features + timestep
        self.time_embed = SinusoidalPositionEmbeddings(d_model)
        self.denoiser = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=n_layers
        )
        self.mask_in = nn.Linear(1, d_model)
        self.mask_out = nn.Linear(d_model, 1)
    
    def forward(self, text, noisy_mask, t):
        # Encode text
        text_feat = self.text_encoder(text)  # [B, L, D]
        
        # Embed noisy mask
        mask_feat = self.mask_in(noisy_mask.unsqueeze(-1))  # [B, L, D]
        
        # Add time embedding
        t_emb = self.time_embed(t)  # [B, D]
        
        # Combine: text + mask + time
        x = text_feat + mask_feat + t_emb.unsqueeze(1)
        
        # Denoise
        x = self.denoiser(x)
        
        # Predict clean mask
        return self.mask_out(x).squeeze(-1)  # [B, L]
```

### For Speed: Replace Transformer with Mamba

```python
# Swap TransformerEncoder for Mamba
from mamba_ssm import Mamba

self.denoiser = nn.Sequential(*[
    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
    for _ in range(n_layers)
])
```

Benefits:
- Linear complexity in sequence length
- Faster inference
- Better for long sequences

## Recommended Approach for Secret Detection

### Architecture

1. **Text Encoder**: Byte-level embeddings + conv layers (reuse existing line filter encoder)
2. **Denoiser**: Small Mamba stack (4-6 layers) for speed and long-sequence support
3. **Training**: Rectified flow (simpler than score matching, naturally few-step)
4. **Inference**: 2-4 steps with consistency-style refinement

### Training Recipe

```python
def train_step(model, text, true_mask):
    # Rectified flow: interpolate between noise and target
    t = torch.rand(text.shape[0], 1, 1)
    noise = torch.randn_like(true_mask.float())
    noisy_mask = t * true_mask + (1 - t) * noise
    
    # Predict velocity (direction from noise to clean)
    v_pred = model(text, noisy_mask, t.squeeze())
    v_target = true_mask - noise
    
    return F.mse_loss(v_pred, v_target)
```

### Inference

```python
@torch.no_grad()
def predict_mask(model, text, steps=4):
    mask = torch.randn(text.shape[0], text.shape[1])
    
    for i in range(steps):
        t = torch.tensor([1 - i / steps])
        v = model(text, mask, t)
        mask = mask + v / steps  # Euler step
    
    return (mask > 0.5).float()
```

## Speed Benchmarks (Expected)

| Method | Steps | Relative Speed |
|--------|-------|----------------|
| Standard DDPM | 1000 | 1x |
| DDIM | 50 | 20x |
| Rectified Flow | 10 | 100x |
| Consistency Model | 1-2 | 500-1000x |
| Our target | 2-4 | 250-500x |

## References

1. Song et al. "Consistency Models" (2023)
2. Lipman et al. "Flow Matching for Generative Modeling" (2023)
3. Liu et al. "Flow Straight and Fast" (2023) - Rectified Flow
4. Luo et al. "Latent Consistency Models" (2023)
5. Salimans & Ho "Progressive Distillation" (2022)
6. Austin et al. "D3PM: Structured Denoising Diffusion Models" (2021)
7. Wu et al. "DiffuMask" (2023)
