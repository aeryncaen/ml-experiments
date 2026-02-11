"""ByteChunkVAE — learns to compress fixed-size byte chunks into continuous latents."""

from .model import ByteChunkVAE, VAEConfig

__all__ = ["ByteChunkVAE", "VAEConfig"]
