"""
R-JEPA (Reasoning Joint Embedding Predictive Architecture).

World model for text reasoning in latent space.
Adapted from V-JEPA (Meta AI) for 1D reasoning sequences.
"""

# Core V-JEPA adapted modules
from .model import ReasoningJEPA, reasoning_jepa_base, RJEPA_CONFIGS
from .step_transformer import (
    StepTransformer,
    step_transformer_tiny,
    step_transformer_small,
    step_transformer_base,
    step_transformer_large,
    step_transformer_huge,
    STEP_TRANSFORMER_EMBED_DIMS,
)
from .step_predictor import StepPredictor, step_predictor
from .multiblock1d import MaskCollator
from .modules import Block, Attention, MLP
from .pos_embs import get_1d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

# Legacy modules (if they exist)
try:
    from .dataset import LatentDataset
except ImportError:
    LatentDataset = None

try:
    from .trainer import RJEPATrainer
except ImportError:
    RJEPATrainer = None

try:
    from .client import RJEPAClient
except ImportError:
    RJEPAClient = None

try:
    from .service import create_app, RJEPAService
except ImportError:
    create_app = None
    RJEPAService = None

__all__ = [
    # Main model
    "ReasoningJEPA",
    "reasoning_jepa_base",
    "RJEPA_CONFIGS",
    # Encoder variants
    "StepTransformer",
    "step_transformer_tiny",
    "step_transformer_small",
    "step_transformer_base",
    "step_transformer_large",
    "step_transformer_huge",
    "STEP_TRANSFORMER_EMBED_DIMS",
    # Predictor
    "StepPredictor",
    "step_predictor",
    # Masking
    "MaskCollator",
    # Core modules
    "Block",
    "Attention",
    "MLP",
    # Positional embeddings
    "get_1d_sincos_pos_embed",
    "get_1d_sincos_pos_embed_from_grid",
    # Legacy (if available)
    "LatentDataset",
    "RJEPATrainer",
    "RJEPAClient",
    "create_app",
    "RJEPAService",
]
