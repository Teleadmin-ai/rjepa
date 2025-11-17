"""
Evaluation metrics for R-JEPA.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def extract_answer(text: str, answer_type: str = "numeric") -> Optional[str]:
    """
    Extract final answer from generated text.

    Args:
        text: Generated response text
        answer_type: Type of answer to extract ("numeric", "boolean", "text")

    Returns:
        Extracted answer or None if not found
    """
    if answer_type == "numeric":
        # Try to find last number in text
        # Look for patterns like "answer is 42", "= 42", "42."
        patterns = [
            r"(?:answer is|answer:|=)\s*([+-]?\d+(?:\.\d+)?)",
            r"(?:final answer|result):\s*([+-]?\d+(?:\.\d+)?)",
            r"([+-]?\d+(?:\.\d+)?)\s*$",  # Last number in text
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)

        # Fallback: find any number
        numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]  # Return last number

        return None

    elif answer_type == "boolean":
        text_lower = text.lower()
        if "true" in text_lower or "yes" in text_lower:
            return "true"
        elif "false" in text_lower or "no" in text_lower:
            return "false"
        return None

    else:  # text
        # Return last sentence or last line
        sentences = text.strip().split(".")
        return sentences[-1].strip() if sentences else None


def compute_accuracy(
    predictions: List[str],
    targets: List[str],
    answer_type: str = "numeric",
    tolerance: float = 1e-5,
) -> float:
    """
    Compute accuracy of predictions.

    Args:
        predictions: List of predicted answers
        targets: List of ground truth answers
        answer_type: Type of answer ("numeric", "boolean", "text")
        tolerance: Tolerance for numeric comparison

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = 0
    for pred, target in zip(predictions, targets):
        if answer_type == "numeric":
            try:
                pred_num = float(pred) if pred else None
                target_num = float(target)
                if pred_num is not None and abs(pred_num - target_num) < tolerance:
                    correct += 1
            except (ValueError, TypeError):
                continue
        else:
            if pred and pred.lower().strip() == target.lower().strip():
                correct += 1

    accuracy = correct / len(predictions)
    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")

    return accuracy


def compute_pass_at_k(
    results: List[bool],
    k: int = 1,
    n: int = None,
) -> float:
    """
    Compute pass@k metric (used for code generation).

    Pass@k = probability that at least one of k samples passes.

    Args:
        results: List of boolean results (True = pass, False = fail)
        k: Number of samples to consider
        n: Total number of samples (defaults to len(results))

    Returns:
        Pass@k score (0.0 to 1.0)
    """
    if n is None:
        n = len(results)

    if n < k:
        return 0.0

    # Count number of passes
    num_passes = sum(results)

    if num_passes == 0:
        return 0.0

    # Compute pass@k using combinatorial formula
    # pass@k = 1 - (n-c choose k) / (n choose k)
    # where c = number of correct samples

    from math import comb

    if num_passes >= k:
        return 1.0

    pass_at_k = 1.0 - (comb(n - num_passes, k) / comb(n, k))

    return pass_at_k


def compute_correlation(
    jepa_scores: List[float],
    correctness: List[bool],
    method: str = "pearson",
) -> Tuple[float, float]:
    """
    Compute correlation between JEPA scores and correctness.

    IMPORTANT: Lower JEPA-loss should correlate with higher correctness.
    We expect negative correlation (low loss = correct, high loss = incorrect).

    Args:
        jepa_scores: List of JEPA scores (losses)
        correctness: List of boolean correctness values
        method: Correlation method ("pearson" or "spearman")

    Returns:
        (correlation coefficient, p-value)
    """
    if len(jepa_scores) != len(correctness):
        raise ValueError("JEPA scores and correctness must have same length")

    if len(jepa_scores) < 2:
        return 0.0, 1.0

    # Convert correctness to numeric (1.0 = correct, 0.0 = incorrect)
    correctness_numeric = np.array([1.0 if c else 0.0 for c in correctness])
    jepa_scores_array = np.array(jepa_scores)

    # Compute correlation
    if method == "pearson":
        corr, pval = pearsonr(jepa_scores_array, correctness_numeric)
    elif method == "spearman":
        corr, pval = spearmanr(jepa_scores_array, correctness_numeric)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    logger.info(f"{method.capitalize()} correlation: {corr:.4f} (p={pval:.4e})")

    return corr, pval


def compute_metrics_summary(
    results: List[Dict[str, Any]],
    include_correlation: bool = True,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics from evaluation results.

    Args:
        results: List of evaluation results, each with:
            - prediction: str
            - target: str
            - is_correct: bool
            - jepa_loss: float (optional)
        include_correlation: Whether to compute JEPA-correctness correlation

    Returns:
        Dictionary of metrics
    """
    if not results:
        return {"accuracy": 0.0, "num_samples": 0}

    # Extract data
    predictions = [r["prediction"] for r in results]
    targets = [r["target"] for r in results]
    correctness = [r.get("is_correct", False) for r in results]
    jepa_losses = [r.get("jepa_loss") for r in results if "jepa_loss" in r]

    # Compute accuracy
    accuracy = sum(correctness) / len(correctness)

    metrics = {
        "accuracy": accuracy,
        "num_samples": len(results),
        "num_correct": sum(correctness),
        "num_incorrect": len(correctness) - sum(correctness),
    }

    # Compute JEPA statistics
    if jepa_losses:
        jepa_losses_valid = [j for j in jepa_losses if j is not None]
        if jepa_losses_valid:
            metrics["jepa_loss_mean"] = float(np.mean(jepa_losses_valid))
            metrics["jepa_loss_std"] = float(np.std(jepa_losses_valid))
            metrics["jepa_loss_min"] = float(np.min(jepa_losses_valid))
            metrics["jepa_loss_max"] = float(np.max(jepa_losses_valid))

            # Separate by correctness
            correct_losses = [r["jepa_loss"] for r in results
                            if r.get("is_correct") and "jepa_loss" in r]
            incorrect_losses = [r["jepa_loss"] for r in results
                              if not r.get("is_correct") and "jepa_loss" in r]

            if correct_losses:
                metrics["jepa_loss_correct_mean"] = float(np.mean(correct_losses))
            if incorrect_losses:
                metrics["jepa_loss_incorrect_mean"] = float(np.mean(incorrect_losses))

            # Compute correlation
            if include_correlation and len(jepa_losses_valid) >= 2:
                corr, pval = compute_correlation(jepa_losses_valid, correctness[:len(jepa_losses_valid)])
                metrics["jepa_correctness_correlation"] = corr
                metrics["jepa_correctness_pvalue"] = pval

    return metrics
