"""
R-JEPA (Reasoning Joint Embedding Predictive Architecture).

World model for text reasoning in latent space.
"""
from .model import ReasoningJEPA, create_rjepa_model
from .encoder import ReasoningEncoder
from .predictor import ReasoningPredictor
from .maskers import (
    RandomMasker,
    ContiguousMasker,
    HierarchicalMasker,
    MaskCollator,
    create_masker,
)
from .losses import JEPALoss
from .dataset import LatentDataset

__all__ = [
    "ReasoningJEPA",
    "create_rjepa_model",
    "ReasoningEncoder",
    "ReasoningPredictor",
    "RandomMasker",
    "ContiguousMasker",
    "HierarchicalMasker",
    "MaskCollator",
    "create_masker",
    "JEPALoss",
    "LatentDataset",
]
