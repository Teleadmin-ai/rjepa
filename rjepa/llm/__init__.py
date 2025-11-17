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
from .projections import (
    LatentProjector,
    MultiLLMAdapter,
    create_adapter_for_llm,
    save_adapter,
    load_adapter,
    AdapterTrainer,
    LLM_HIDDEN_SIZES,
)

__all__ = [
    # Core LLM adapter
    "LLMAdapter",
    # Step segmentation
    "segment_by_step_markers",
    "segment_by_sentences",
    "segment_by_connectors",
    "segment_auto",
    # Quantization utils
    "check_quantization_available",
    "get_quantization_config",
    "estimate_vram_usage",
    "recommend_quantization",
    # Multi-LLM projections (Phase 16)
    "LatentProjector",
    "MultiLLMAdapter",
    "create_adapter_for_llm",
    "save_adapter",
    "load_adapter",
    "AdapterTrainer",
    "LLM_HIDDEN_SIZES",
]
