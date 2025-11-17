"""
Test Teacher components.
"""
import pytest
from unittest.mock import Mock, patch

from rjepa.teacher import (
    TeacherClient,
    MultiSourceTeacher,
    BudgetTracker,
    Validator,
    MathValidator,
)
from rjepa.data.schemas import Problem, ChainOfThought


@pytest.fixture
def mock_teacher_client():
    """Mock teacher client for testing."""
    client = Mock(spec=TeacherClient)
    client.model = "test-model"
    client.base_url = "http://localhost:8001/v1"
    client.generate.return_value = "This is a test response"
    client.generate_json.return_value = {"answer": "42"}
    client.count_tokens.return_value = 100
    return client


def test_budget_tracker():
    """Test budget tracker."""
    tracker = BudgetTracker(max_budget_usd=10.0)

    # Record usage
    tracker.record_usage("gpt-4-turbo", input_tokens=1000, output_tokens=500)

    # Check cost
    assert tracker.get_total_cost() > 0
    assert tracker.get_total_cost() < 10.0  # Should be well under budget

    # Check summary
    summary = tracker.get_usage_summary()
    assert "gpt-4-turbo" in summary["models"]
    assert summary["total_cost"] > 0


def test_budget_tracker_exceeded():
    """Test budget exceeded detection."""
    tracker = BudgetTracker(max_budget_usd=0.001)  # Very low budget

    # Record large usage
    tracker.record_usage("gpt-4", input_tokens=100000, output_tokens=50000)

    # Should be exceeded
    assert tracker.is_budget_exceeded()
    assert tracker.get_budget_remaining() == 0.0


def test_math_validator():
    """Test math validator."""
    validator = MathValidator()

    # Create test problem
    problem = Problem(
        problem_id="test_math_1",
        domain="math",
        subdomain="arithmetic",
        source="test",
        difficulty="easy",
        statement="What is 2 + 2?",
        answer_gold="4",
    )

    # Create correct CoT
    correct_cot = ChainOfThought(
        cot_id="test_cot_1",
        problem_id="test_math_1",
        steps=["Step 1: Add 2 and 2", "Step 2: Get 4"],
        final_answer="4",
        is_valid=False,
        validation_reason="",
        teacher_model="test",
        source="test",
    )

    # Validate
    is_valid, reason = validator.validate(problem, correct_cot)

    assert is_valid
    assert "match" in reason.lower()


def test_math_validator_incorrect():
    """Test math validator with incorrect answer."""
    validator = MathValidator()

    # Create test problem
    problem = Problem(
        problem_id="test_math_2",
        domain="math",
        subdomain="arithmetic",
        source="test",
        difficulty="easy",
        statement="What is 2 + 2?",
        answer_gold="4",
    )

    # Create incorrect CoT
    incorrect_cot = ChainOfThought(
        cot_id="test_cot_2",
        problem_id="test_math_2",
        steps=["Step 1: Add 2 and 2", "Step 2: Get 5"],
        final_answer="5",
        is_valid=False,
        validation_reason="",
        teacher_model="test",
        source="test",
    )

    # Validate
    is_valid, reason = validator.validate(problem, incorrect_cot)

    assert not is_valid
    assert "mismatch" in reason.lower()


def test_multi_source_teacher(mock_teacher_client):
    """Test multi-source teacher."""
    teacher = MultiSourceTeacher()

    # Add clients
    client1 = mock_teacher_client
    client2 = Mock(spec=TeacherClient)
    client2.model = "test-model-2"
    client2.generate.return_value = "Another response"
    client2.count_tokens.return_value = 50

    teacher.add_client("model1", client1)
    teacher.add_client("model2", client2)

    # Generate diverse
    results = teacher.generate_diverse("Test prompt", num_per_source=2)

    assert len(results) == 4  # 2 sources x 2 samples
    assert any(r["source"] == "model1" for r in results)
    assert any(r["source"] == "model2" for r in results)


def test_validator_dispatch():
    """Test validator domain dispatch."""
    validator = Validator()

    # Math problem
    math_problem = Problem(
        problem_id="test_1",
        domain="math",
        subdomain="algebra",
        source="test",
        difficulty="medium",
        statement="Solve x + 2 = 5",
        answer_gold="3",
    )

    math_cot = ChainOfThought(
        cot_id="test_cot_1",
        problem_id="test_1",
        steps=["Step 1: Subtract 2 from both sides", "Step 2: x = 3"],
        final_answer="3",
        is_valid=False,
        validation_reason="",
        teacher_model="test",
        source="test",
    )

    # Should dispatch to math validator
    is_valid, reason = validator.validate(math_problem, math_cot)
    assert is_valid
