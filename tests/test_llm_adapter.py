"""
Test LLM Adapter.
"""
import pytest
import torch
from rjepa.llm import LLMAdapter


@pytest.fixture(scope="module")
def llm_adapter():
    """Load a small test model (GPT-2 for speed)."""
    # Use GPT-2 for testing (small, fast, no quantization needed)
    return LLMAdapter(
        model_name="gpt2",
        quantization=None,
        layer_to_extract=-2,
    )


def test_adapter_loads(llm_adapter):
    """Test that adapter loads successfully."""
    assert llm_adapter is not None
    assert llm_adapter.model is not None
    assert llm_adapter.tokenizer is not None
    assert llm_adapter.hidden_size > 0
    assert llm_adapter.num_layers > 0


def test_generate_with_cot(llm_adapter):
    """Test CoT generation."""
    results = llm_adapter.generate_with_cot(
        prompt="What is 2 + 2?",
        max_new_tokens=50,
        temperature=0.7,
        num_samples=1,
    )

    assert len(results) == 1
    result = results[0]

    assert "full_text" in result
    assert "steps" in result
    assert "tokens" in result
    assert "step_boundaries" in result

    assert isinstance(result["full_text"], str)
    assert isinstance(result["steps"], list)
    assert isinstance(result["tokens"], torch.Tensor)
    assert isinstance(result["step_boundaries"], list)


def test_extract_latents(llm_adapter):
    """Test latent extraction."""
    # Generate first
    results = llm_adapter.generate_with_cot(
        prompt="Count to 3.",
        max_new_tokens=30,
        num_samples=1,
    )
    result = results[0]

    # Extract latents
    latents = llm_adapter.extract_latents(
        tokens=result["tokens"],
        step_boundaries=result["step_boundaries"],
    )

    # Check shape
    assert latents.dim() == 2  # [num_steps, hidden_size]
    assert latents.shape[0] == len(result["steps"])  # num_steps
    assert latents.shape[1] == llm_adapter.hidden_size  # hidden_size

    # Check dtype
    assert latents.dtype in [torch.float16, torch.bfloat16, torch.float32]


def test_multiple_samples(llm_adapter):
    """Test generating multiple samples."""
    results = llm_adapter.generate_with_cot(
        prompt="What is 5 + 5?",
        max_new_tokens=30,
        temperature=0.8,
        num_samples=3,
    )

    assert len(results) == 3

    for result in results:
        assert "full_text" in result
        assert len(result["steps"]) >= 1


def test_step_segmentation(llm_adapter):
    """Test step segmentation logic."""
    # Force structured output
    results = llm_adapter.generate_with_cot(
        prompt="Solve: x + 2 = 5",
        max_new_tokens=100,
        force_structure=True,
        num_samples=1,
    )

    result = results[0]

    # Should have segmented steps (at least fallback to 1 step)
    assert len(result["steps"]) >= 1

    # Should have corresponding boundaries
    assert len(result["step_boundaries"]) == len(result["steps"])


def test_repr(llm_adapter):
    """Test __repr__ method."""
    repr_str = repr(llm_adapter)
    assert "LLMAdapter" in repr_str
    assert "model=" in repr_str
    assert "hidden_size=" in repr_str
