"""
Tests for R-JEPA evaluation framework.
"""
import pytest
from typing import List, Dict, Any


# ============================================================================
# Test Metrics
# ============================================================================

def test_extract_answer_numeric():
    """Test numeric answer extraction."""
    from rjepa.evaluation import extract_answer

    # Test various numeric patterns
    test_cases = [
        ("The answer is 42", "42"),
        ("answer: 3.14", "3.14"),
        ("Result = 100", "100"),
        ("Final answer: -25", "-25"),
        ("No number here", None),  # No clear answer pattern
    ]

    for text, expected in test_cases:
        result = extract_answer(text, answer_type="numeric")
        if expected is not None:
            assert result == expected, f"Expected {expected}, got {result} for text: {text}"


def test_extract_answer_boolean():
    """Test boolean answer extraction."""
    from rjepa.evaluation import extract_answer

    test_cases = [
        ("The statement is true", "true"),
        ("This is false", "false"),
        ("Yes, it's correct", "true"),
        ("No, that's wrong", "false"),
    ]

    for text, expected in test_cases:
        result = extract_answer(text, answer_type="boolean")
        assert result == expected, f"Expected {expected}, got {result} for text: {text}"


def test_compute_accuracy():
    """Test accuracy computation."""
    from rjepa.evaluation import compute_accuracy

    # Perfect accuracy
    predictions = ["42", "10", "20"]
    targets = ["42", "10", "20"]
    accuracy = compute_accuracy(predictions, targets, answer_type="numeric")
    assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"

    # 75% accuracy
    predictions = ["42", "10", "20", "42"]
    targets = ["42", "15", "20", "42"]
    accuracy = compute_accuracy(predictions, targets, answer_type="numeric")
    assert accuracy == 0.75, f"Expected 0.75, got {accuracy}"

    # 0% accuracy
    predictions = ["1", "2", "3"]
    targets = ["10", "20", "30"]
    accuracy = compute_accuracy(predictions, targets, answer_type="numeric")
    assert accuracy == 0.0, f"Expected 0.0, got {accuracy}"


def test_compute_pass_at_k():
    """Test pass@k metric."""
    from rjepa.evaluation import compute_pass_at_k

    # All pass
    results = [True, True, True, True, True]
    pass_at_1 = compute_pass_at_k(results, k=1)
    assert pass_at_1 == 1.0, f"Expected 1.0, got {pass_at_1}"

    # None pass
    results = [False, False, False, False, False]
    pass_at_1 = compute_pass_at_k(results, k=1)
    assert pass_at_1 == 0.0, f"Expected 0.0, got {pass_at_1}"

    # Mixed
    results = [True, False, True, False, True]
    pass_at_1 = compute_pass_at_k(results, k=1)
    assert 0.0 < pass_at_1 < 1.0, f"Expected value between 0 and 1, got {pass_at_1}"


def test_compute_correlation():
    """Test correlation computation."""
    from rjepa.evaluation import compute_correlation

    # Perfect negative correlation (low loss = correct)
    jepa_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    correctness = [True, True, False, False, False]

    corr, pval = compute_correlation(jepa_scores, correctness, method="pearson")
    assert corr < 0, f"Expected negative correlation, got {corr}"

    # Test Spearman
    corr_spear, pval_spear = compute_correlation(jepa_scores, correctness, method="spearman")
    assert corr_spear < 0, f"Expected negative Spearman correlation, got {corr_spear}"


def test_compute_metrics_summary():
    """Test comprehensive metrics summary."""
    from rjepa.evaluation import compute_metrics_summary

    results = [
        {"prediction": "42", "target": "42", "is_correct": True, "jepa_loss": 0.1},
        {"prediction": "10", "target": "15", "is_correct": False, "jepa_loss": 0.3},
        {"prediction": "20", "target": "20", "is_correct": True, "jepa_loss": 0.15},
        {"prediction": "5", "target": "5", "is_correct": True, "jepa_loss": 0.12},
    ]

    metrics = compute_metrics_summary(results, include_correlation=True)

    # Check basic metrics
    assert "accuracy" in metrics
    assert "num_samples" in metrics
    assert "num_correct" in metrics
    assert "num_incorrect" in metrics

    # Check JEPA stats
    assert "jepa_loss_mean" in metrics
    assert "jepa_loss_std" in metrics
    assert "jepa_loss_correct_mean" in metrics
    assert "jepa_loss_incorrect_mean" in metrics

    # Check correlation
    assert "jepa_correctness_correlation" in metrics
    assert "jepa_correctness_pvalue" in metrics

    # Verify accuracy
    assert metrics["accuracy"] == 0.75, f"Expected 0.75, got {metrics['accuracy']}"
    assert metrics["num_correct"] == 3
    assert metrics["num_incorrect"] == 1


# ============================================================================
# Test Benchmarks (mocked, no actual dataset loading)
# ============================================================================

def test_benchmark_functions_exist():
    """Test that benchmark loading functions exist."""
    from rjepa.evaluation import load_gsm8k, load_math, load_humaneval, create_mini_benchmark

    # Just check they're callable (don't actually load datasets)
    assert callable(load_gsm8k)
    assert callable(load_math)
    assert callable(load_humaneval)
    assert callable(create_mini_benchmark)


# ============================================================================
# Test A/B Testing
# ============================================================================

def test_run_ab_test_mock():
    """Test A/B testing framework with mock functions."""
    from rjepa.evaluation import run_ab_test

    problems = [
        {"problem_id": "p1", "question": "What is 2+2?", "answer": "4"},
        {"problem_id": "p2", "question": "What is 3*3?", "answer": "9"},
    ]

    def baseline_fn(question):
        return "The answer is 4" if "2+2" in question else "The answer is 9"

    def treatment_fn(question):
        return "The answer is 4" if "2+2" in question else "The answer is 9"

    def answer_extractor(response):
        import re
        match = re.search(r"answer is (\d+)", response)
        return match.group(1) if match else None

    results = run_ab_test(
        problems=problems,
        baseline_fn=baseline_fn,
        treatment_fn=treatment_fn,
        answer_extractor=answer_extractor,
        max_samples=2,
    )

    # Check results structure
    assert "baseline_accuracy" in results
    assert "treatment_accuracy" in results
    assert "delta" in results
    assert "relative_improvement" in results
    assert "num_samples" in results
    assert "baseline_results" in results
    assert "treatment_results" in results

    # Both should get 100% accuracy
    assert results["baseline_accuracy"] == 1.0
    assert results["treatment_accuracy"] == 1.0
    assert results["delta"] == 0.0


# ============================================================================
# Test Visualization (check they're callable, don't generate plots)
# ============================================================================

def test_visualization_functions_exist():
    """Test that visualization functions exist."""
    from rjepa.evaluation import (
        plot_jepa_loss_distribution,
        plot_correlation_scatter,
        plot_accuracy_comparison,
        plot_mode_comparison,
        plot_jepa_loss_by_domain,
        generate_evaluation_report,
    )

    # Just check they're callable (don't actually generate plots)
    assert callable(plot_jepa_loss_distribution)
    assert callable(plot_correlation_scatter)
    assert callable(plot_accuracy_comparison)
    assert callable(plot_mode_comparison)
    assert callable(plot_jepa_loss_by_domain)
    assert callable(generate_evaluation_report)


# ============================================================================
# Integration Test (mocked components)
# ============================================================================

def test_evaluation_workflow_mock():
    """Test complete evaluation workflow with mocked components."""
    from rjepa.evaluation import compute_accuracy, compute_metrics_summary

    # Simulate baseline results
    baseline_results = [
        {"prediction": "42", "target": "42", "is_correct": True},
        {"prediction": "10", "target": "15", "is_correct": False},
        {"prediction": "20", "target": "20", "is_correct": True},
    ]

    # Simulate JEPA results (improved)
    jepa_results = [
        {"prediction": "42", "target": "42", "is_correct": True, "jepa_loss": 0.1},
        {"prediction": "15", "target": "15", "is_correct": True, "jepa_loss": 0.2},
        {"prediction": "20", "target": "20", "is_correct": True, "jepa_loss": 0.15},
    ]

    # Compute metrics
    baseline_metrics = compute_metrics_summary(baseline_results, include_correlation=False)
    jepa_metrics = compute_metrics_summary(jepa_results, include_correlation=True)

    # Verify improvement
    assert baseline_metrics["accuracy"] == 2/3
    assert jepa_metrics["accuracy"] == 1.0
    assert jepa_metrics["accuracy"] > baseline_metrics["accuracy"]

    # Verify JEPA-specific metrics exist
    assert "jepa_loss_mean" in jepa_metrics
    assert "jepa_correctness_correlation" in jepa_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
