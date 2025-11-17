"""
Quantization utilities for LLM loading.

Provides helpers for AWQ, GPTQ, and bitsandbytes quantization.
"""
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def check_quantization_available(quant_type: str) -> bool:
    """
    Check if a quantization backend is available.

    Args:
        quant_type: "awq", "gptq", or "bitsandbytes"

    Returns:
        True if available
    """
    if quant_type == "awq":
        try:
            import awq
            return True
        except ImportError:
            return False

    elif quant_type == "gptq":
        try:
            import auto_gptq
            return True
        except ImportError:
            return False

    elif quant_type == "bitsandbytes":
        try:
            import bitsandbytes
            return True
        except ImportError:
            return False

    return False


def get_quantization_config(quant_type: str) -> Dict[str, Any]:
    """
    Get default quantization config for a given type.

    Args:
        quant_type: "awq-4bit", "gptq-4bit", "bnb-4bit", "bnb-8bit"

    Returns:
        Config dict for transformers
    """
    if quant_type == "awq-4bit":
        return {
            "fuse_layers": True,
            "device_map": "auto",
        }

    elif quant_type == "gptq-4bit":
        return {
            "device": "cuda:0",
            "use_safetensors": True,
        }

    elif quant_type == "bnb-4bit":
        from transformers import BitsAndBytesConfig
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ),
            "device_map": "auto",
        }

    elif quant_type == "bnb-8bit":
        from transformers import BitsAndBytesConfig
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
            ),
            "device_map": "auto",
        }

    raise ValueError(f"Unknown quantization type: {quant_type}")


def estimate_vram_usage(
    num_params_billions: float,
    quant_type: Optional[str] = None,
    batch_size: int = 1,
    seq_length: int = 2048
) -> float:
    """
    Estimate VRAM usage for a model.

    Args:
        num_params_billions: Number of params in billions
        quant_type: Quantization type (None = fp16)
        batch_size: Batch size
        seq_length: Sequence length

    Returns:
        Estimated VRAM in GB
    """
    # Base model size
    if quant_type is None or quant_type == "fp16":
        model_size_gb = num_params_billions * 2  # 2 bytes per param
    elif "4bit" in quant_type:
        model_size_gb = num_params_billions * 0.5  # 0.5 bytes per param
    elif "8bit" in quant_type:
        model_size_gb = num_params_billions * 1  # 1 byte per param
    else:
        model_size_gb = num_params_billions * 2

    # KV cache (rough estimate)
    kv_cache_gb = (batch_size * seq_length * 4096 * 2 * 2) / 1e9  # rough

    # Activations (rough estimate)
    activations_gb = (batch_size * seq_length * 4096 * 4) / 1e9

    total_gb = model_size_gb + kv_cache_gb + activations_gb

    logger.info(f"VRAM estimate for {num_params_billions}B model ({quant_type}):")
    logger.info(f"  Model: {model_size_gb:.2f} GB")
    logger.info(f"  KV cache: {kv_cache_gb:.2f} GB")
    logger.info(f"  Activations: {activations_gb:.2f} GB")
    logger.info(f"  Total: {total_gb:.2f} GB")

    return total_gb


def recommend_quantization(available_vram_gb: float, model_name: str) -> Optional[str]:
    """
    Recommend quantization based on available VRAM.

    Args:
        available_vram_gb: Available VRAM in GB
        model_name: Model name (to infer size)

    Returns:
        Recommended quantization type or None
    """
    # Infer model size from name
    if "8B" in model_name or "8b" in model_name:
        num_params = 8
    elif "13B" in model_name or "13b" in model_name:
        num_params = 13
    elif "32B" in model_name or "32b" in model_name:
        num_params = 32
    elif "70B" in model_name or "70b" in model_name:
        num_params = 70
    else:
        logger.warning(f"Could not infer model size from name: {model_name}")
        return None

    # Estimate required VRAM for different quants
    fp16_vram = estimate_vram_usage(num_params, None)
    awq4_vram = estimate_vram_usage(num_params, "awq-4bit")
    bnb8_vram = estimate_vram_usage(num_params, "bnb-8bit")

    # Recommend based on available VRAM
    if available_vram_gb >= fp16_vram:
        return None  # Can load in fp16
    elif available_vram_gb >= bnb8_vram:
        return "bnb-8bit"
    elif available_vram_gb >= awq4_vram:
        return "awq-4bit"
    else:
        logger.error(f"Insufficient VRAM ({available_vram_gb} GB) for {model_name}")
        return None
