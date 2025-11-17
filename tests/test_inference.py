"""
Test R-JEPA Inference Modes (rerank, nudge, plan).
"""
import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from rjepa.inference import (
    rerank_cots_with_jepa,
    rerank_existing_cots,
    nudge_reasoning_stepwise,
    nudge_with_regeneration,
    complete_reasoning_plan,
)


@pytest.fixture
def mock_llm():
    """Create mock LLM adapter."""
    llm = Mock()
    llm.layer_to_extract = -2

    # Mock tokenizer
    llm.tokenizer = Mock()
    llm.tokenizer.encode = Mock(return_value=torch.randint(0, 1000, (1, 50)))

    # Mock generate_with_cot
    def mock_generate(prompt, **kwargs):
        num_samples = kwargs.get("num_samples", 1)
        results = []
        for i in range(num_samples):
            results.append({
                "full_text": f"Step 1: First reasoning\nStep 2: Second reasoning\nStep 3: Final answer is 42",
                "steps": [
                    "Step 1: First reasoning",
                    "Step 2: Second reasoning",
                    "Step 3: Final answer is 42",
                ],
                "tokens": torch.randint(0, 1000, (1, 50)),
                "step_boundaries": [(0, 16), (16, 32), (32, 50)],
            })
        return results

    llm.generate_with_cot = Mock(side_effect=mock_generate)

    # Mock extract_latents
    def mock_extract(tokens, layer_idx, step_boundaries):
        num_steps = len(step_boundaries)
        return torch.randn(num_steps, 64)

    llm.extract_latents = Mock(side_effect=mock_extract)

    return llm


@pytest.fixture
def mock_rjepa_client():
    """Create mock R-JEPA client."""
    client = Mock()

    # Mock score
    def mock_score(latents, **kwargs):
        return {
            "jepa_loss": 0.5,
            "num_steps": latents.shape[0],
            "num_masked": int(latents.shape[0] * 0.5),
            "device": "cpu",
        }

    client.score = Mock(side_effect=mock_score)

    # Mock predict_masked
    def mock_predict(latents, mask_indices, **kwargs):
        num_masked = len(mask_indices)
        hidden_dim = latents.shape[1]
        return torch.randn(num_masked, hidden_dim)

    client.predict_masked = Mock(side_effect=mock_predict)

    return client


def test_rerank_cots_with_jepa(mock_llm, mock_rjepa_client):
    """Test re-ranking CoT candidates."""
    result = rerank_cots_with_jepa(
        prompt="What is 2+2?",
        llm=mock_llm,
        rjepa_client=mock_rjepa_client,
        num_samples=3,
        temperature=0.8,
    )

    assert "best_cot" in result
    assert "candidates" in result
    assert "num_candidates" in result

    assert result["num_candidates"] == 3
    assert len(result["candidates"]) == 3

    # Best cot should have all fields
    best_cot = result["best_cot"]
    assert "full_text" in best_cot
    assert "steps" in best_cot
    assert "score" in best_cot
    assert "jepa_loss" in best_cot

    # Candidates should be sorted by score
    scores = [c["score"] for c in result["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_rerank_existing_cots(mock_llm, mock_rjepa_client):
    """Test re-ranking existing CoT candidates."""
    cots = [
        "Step 1: First approach\nStep 2: Result is 4",
        "Step 1: Second approach\nStep 2: Result is 5",
        "Step 1: Third approach\nStep 2: Result is 4",
    ]

    result = rerank_existing_cots(
        cots=cots,
        llm=mock_llm,
        rjepa_client=mock_rjepa_client,
        mask_ratio=0.5,
    )

    assert "best_cot" in result
    assert "best_score" in result
    assert "candidates" in result

    assert len(result["candidates"]) == 3
    assert result["best_cot"] in cots


def test_nudge_reasoning_stepwise(mock_llm, mock_rjepa_client):
    """Test nudge mode (stepwise correction)."""
    result = nudge_reasoning_stepwise(
        prompt="Solve x + 3 = 7",
        llm=mock_llm,
        rjepa_client=mock_rjepa_client,
        max_steps=5,
        lambda_nudge=0.2,
    )

    assert "full_text" in result
    assert "steps" in result
    assert "corrected" in result
    assert "num_steps" in result

    # Should have step info
    assert len(result["steps"]) > 0
    for step_info in result["steps"]:
        assert "step_idx" in step_info
        assert "text" in step_info


def test_nudge_with_regeneration(mock_llm, mock_rjepa_client):
    """Test nudge with regeneration."""
    result = nudge_with_regeneration(
        prompt="What is 5 * 6?",
        llm=mock_llm,
        rjepa_client=mock_rjepa_client,
        max_attempts=3,
        jepa_threshold=0.5,
    )

    assert "full_text" in result
    assert "steps" in result
    assert "iterations" in result
    assert "final_jepa_loss" in result

    # Should have tried at least once
    assert result["iterations"] >= 1
    assert result["iterations"] <= 3


def test_complete_reasoning_plan(mock_llm, mock_rjepa_client):
    """Test plan completion."""
    partial_steps = [
        "Step 1: Start with equation",
        None,  # Missing
        None,  # Missing
        "Step 4: Final answer",
    ]

    missing_indices = [1, 2]

    result = complete_reasoning_plan(
        partial_steps=partial_steps,
        missing_indices=missing_indices,
        total_steps=4,
        llm=mock_llm,
        rjepa_client=mock_rjepa_client,
    )

    assert "completed_steps" in result
    assert "predicted_steps" in result
    assert "full_text" in result

    # Should have completed all steps
    assert len(result["completed_steps"]) == 4
    assert all(step is not None for step in result["completed_steps"])

    # Should have predicted missing steps
    assert len(result["predicted_steps"]) == 2
    assert 1 in result["predicted_steps"]
    assert 2 in result["predicted_steps"]


def test_rerank_with_different_weights(mock_llm, mock_rjepa_client):
    """Test re-ranking with different alpha/beta weights."""
    # High beta (favor JEPA score)
    result_high_beta = rerank_cots_with_jepa(
        prompt="Test",
        llm=mock_llm,
        rjepa_client=mock_rjepa_client,
        num_samples=2,
        alpha=0.1,
        beta=10.0,
    )

    # High alpha (favor logprob)
    result_high_alpha = rerank_cots_with_jepa(
        prompt="Test",
        llm=mock_llm,
        rjepa_client=mock_rjepa_client,
        num_samples=2,
        alpha=10.0,
        beta=0.1,
    )

    # Both should work
    assert "best_cot" in result_high_beta
    assert "best_cot" in result_high_alpha


def test_nudge_with_high_lambda(mock_llm, mock_rjepa_client):
    """Test nudge with high lambda (strong correction)."""
    result = nudge_reasoning_stepwise(
        prompt="Test",
        llm=mock_llm,
        rjepa_client=mock_rjepa_client,
        lambda_nudge=0.9,  # Strong correction
    )

    assert "corrected" in result


def test_complete_with_empty_context(mock_llm, mock_rjepa_client):
    """Test plan completion with minimal context."""
    partial_steps = [None, None, "Step 3: Final"]

    result = complete_reasoning_plan(
        partial_steps=partial_steps,
        missing_indices=[0, 1],
        total_steps=3,
        llm=mock_llm,
        rjepa_client=mock_rjepa_client,
    )

    assert len(result["completed_steps"]) == 3
