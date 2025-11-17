"""
Test data schemas.
"""
import pytest
from datetime import datetime
from rjepa.data import (
    Problem,
    ChainOfThought,
    LatentSequence,
    DatasetVersion,
    RJEPACheckpoint,
)


def test_problem_schema():
    """Test Problem schema."""
    problem = Problem(
        problem_id="math_001",
        domain="math",
        subdomain="algebra",
        source="teacher_claude",
        difficulty="easy",
        statement="Solve for x: 2x + 5 = 13",
        answer_gold="4",
    )

    assert problem.problem_id == "math_001"
    assert problem.domain == "math"
    assert problem.subdomain == "algebra"
    assert problem.difficulty == "easy"
    assert isinstance(problem.created_at, datetime)


def test_chain_of_thought_schema():
    """Test ChainOfThought schema."""
    cot = ChainOfThought(
        cot_id="cot_001",
        problem_id="math_001",
        steps=[
            "Step 1: Subtract 5 from both sides: 2x = 8",
            "Step 2: Divide by 2: x = 4",
        ],
        final_answer="4",
        is_valid=True,
        validation_reason="Verified symbolically",
        teacher_model="claude-3-5-sonnet",
        source="teacher_claude",
    )

    assert cot.cot_id == "cot_001"
    assert len(cot.steps) == 2
    assert cot.is_valid is True
    assert isinstance(cot.created_at, datetime)


def test_latent_sequence_schema():
    """Test LatentSequence schema."""
    latent = LatentSequence(
        latent_id="latent_001",
        problem_id="math_001",
        cot_id="cot_001",
        llm_tag="qwen3-8b-instruct-awq",
        layer_idx=-2,
        hidden_size=4096,
        num_steps=2,
        step_boundaries=[(0, 15), (15, 28)],
        domain="math",
        subdomain="algebra",
    )

    assert latent.latent_id == "latent_001"
    assert latent.layer_idx == -2
    assert latent.hidden_size == 4096
    assert len(latent.step_boundaries) == 2


def test_dataset_version_schema():
    """Test DatasetVersion schema."""
    version = DatasetVersion(
        version="v1.0.0",
        date="2025-01-15",
        num_problems=12000,
        num_cots=36000,
        sources=["teacher_claude", "teacher_gpt"],
        validation_rate=0.89,
    )

    assert version.version == "v1.0.0"
    assert version.num_problems == 12000
    assert version.validation_rate == 0.89


def test_rjepa_checkpoint_schema():
    """Test RJEPACheckpoint schema."""
    checkpoint = RJEPACheckpoint(
        checkpoint_id="rjepa-qwen3-8b-v1.0.0-epoch-10",
        llm_tag="qwen3-8b",
        dataset_version="v1.0.0",
        epoch=10,
        train_loss=0.15,
        val_loss=0.18,
        config_path="configs/rjepa/base.yaml",
        checkpoint_path="data/checkpoints/rjepa-qwen3-8b/checkpoint-epoch-10.pth",
    )

    assert checkpoint.llm_tag == "qwen3-8b"
    assert checkpoint.epoch == 10
    assert checkpoint.train_loss == 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
