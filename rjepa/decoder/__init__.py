"""
R-JEPA Latent Decoder.

Transforms latent representations back to text (like V-JEPA diffusion decoder).
This allows us to:
1. Decode predicted latents to text (plan mode)
2. Generate text conditioned on latent (nudge mode)
3. Keep R-JEPA frozen (world model as ground truth)

Philosophy: The decoder is trained AFTER R-JEPA is frozen, ensuring
the world model captures conceptual structure, not text generation details.
"""
from .latent_decoder import LatentDecoder, LatentDecoderConfig, create_latent_decoder
from .trainer import LatentDecoderTrainer
from .dataset import LatentTextDataset, create_decoder_dataloaders

__all__ = [
    "LatentDecoder",
    "LatentDecoderConfig",
    "create_latent_decoder",
    "LatentDecoderTrainer",
    "LatentTextDataset",
    "create_decoder_dataloaders",
]
