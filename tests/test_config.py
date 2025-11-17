"""
Test configuration settings.
"""
import pytest
from rjepa.config import settings, Settings


def test_settings_load():
    """Test that settings can be loaded."""
    # Should load from .env or use defaults
    assert settings is not None
    assert isinstance(settings, Settings)


def test_settings_defaults():
    """Test default values are set correctly."""
    assert settings.student_model_name == "Qwen/Qwen3-8B-Instruct"
    assert settings.student_quantization == "awq-4bit"
    assert settings.student_layer_to_extract == -2
    assert settings.teacher_max_budget_per_job == 50.0


def test_settings_validation():
    """Test settings validation."""
    # Layer extraction should be negative
    assert settings.student_layer_to_extract < 0

    # Budget should be positive
    assert settings.teacher_max_budget_per_job > 0

    # Device should be valid
    assert settings.device in ["cuda", "cpu", "mps"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
