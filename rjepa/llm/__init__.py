"""
R-JEPA LLM Package.

Provides adapters for student LLM with latent extraction.
"""
from .adapter import LLMAdapter
from .step_segmentation import (
    segment_by_step_markers,
    segment_by_sentences,
    segment_by_connectors,
    segment_auto,
)
from .quant_utils import (
    check_quantization_available,
    get_quantization_config,
    estimate_vram_usage,
    recommend_quantization,
)

__all__ = [
    "LLMAdapter",
    "segment_by_step_markers",
    "segment_by_sentences",
    "segment_by_connectors",
    "segment_auto",
    "check_quantization_available",
    "get_quantization_config",
    "estimate_vram_usage",
    "recommend_quantization",
]
