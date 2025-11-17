"""
A/B testing framework for comparing JEPA modes.
"""
import logging
from typing import List, Dict, Any, Optional, Callable
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_ab_test(
    problems: List[Dict[str, Any]],
    baseline_fn: Callable,
    treatment_fn: Callable,
    answer_extractor: Optional[Callable] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run A/B test comparing baseline vs treatment.

    Args:
        problems: List of problems to evaluate
        baseline_fn: Function for baseline (JEPA off)
        treatment_fn: Function for treatment (JEPA on)
        answer_extractor: Function to extract answer from response
        max_samples: Maximum number of samples to test

    Returns:
        Dictionary with A/B test results
    """
    logger.info(f"Running A/B test on {len(problems)} problems...")

    if max_samples:
        problems = problems[:max_samples]

    baseline_results = []
    treatment_results = []

    for problem in tqdm(problems, desc="A/B Testing"):
        # Run baseline (JEPA off)
        try:
            baseline_response = baseline_fn(problem["question"])
            baseline_answer = answer_extractor(baseline_response) if answer_extractor else baseline_response

            baseline_correct = (baseline_answer == problem["answer"])

            baseline_results.append({
                "problem_id": problem["problem_id"],
                "question": problem["question"],
                "answer": problem["answer"],
                "prediction": baseline_answer,
                "is_correct": baseline_correct,
                "response": baseline_response,
            })

        except Exception as e:
            logger.error(f"Baseline failed for {problem['problem_id']}: {e}")
            baseline_results.append({
                "problem_id": problem["problem_id"],
                "is_correct": False,
                "error": str(e),
            })

        # Run treatment (JEPA on)
        try:
            treatment_response = treatment_fn(problem["question"])
            treatment_answer = answer_extractor(treatment_response) if answer_extractor else treatment_response

            treatment_correct = (treatment_answer == problem["answer"])

            treatment_results.append({
                "problem_id": problem["problem_id"],
                "question": problem["question"],
                "answer": problem["answer"],
                "prediction": treatment_answer,
                "is_correct": treatment_correct,
                "response": treatment_response,
            })

        except Exception as e:
            logger.error(f"Treatment failed for {problem['problem_id']}: {e}")
            treatment_results.append({
                "problem_id": problem["problem_id"],
                "is_correct": False,
                "error": str(e),
            })

    # Compute statistics
    baseline_accuracy = sum(r.get("is_correct", False) for r in baseline_results) / len(baseline_results)
    treatment_accuracy = sum(r.get("is_correct", False) for r in treatment_results) / len(treatment_results)

    delta = treatment_accuracy - baseline_accuracy
    relative_improvement = (delta / baseline_accuracy * 100) if baseline_accuracy > 0 else 0

    logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
    logger.info(f"Treatment accuracy: {treatment_accuracy:.4f}")
    logger.info(f"Delta: {delta:+.4f} ({relative_improvement:+.2f}%)")

    return {
        "baseline_accuracy": baseline_accuracy,
        "treatment_accuracy": treatment_accuracy,
        "delta": delta,
        "relative_improvement": relative_improvement,
        "num_samples": len(problems),
        "baseline_results": baseline_results,
        "treatment_results": treatment_results,
    }


def compare_modes(
    problems: List[Dict[str, Any]],
    llm,
    rjepa_client,
    modes: List[str] = ["off", "rerank", "nudge", "plan"],
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare multiple JEPA modes on same problems.

    Args:
        problems: List of problems to evaluate
        llm: LLM adapter
        rjepa_client: R-JEPA client
        modes: List of modes to compare
        max_samples: Maximum number of samples

    Returns:
        Dictionary with comparison results
    """
    from rjepa.evaluation.metrics import extract_answer
    from rjepa.inference import rerank_cots_with_jepa, nudge_with_regeneration, auto_complete_missing_steps

    logger.info(f"Comparing {len(modes)} modes on {len(problems)} problems...")

    if max_samples:
        problems = problems[:max_samples]

    results_by_mode = {mode: [] for mode in modes}

    for problem in tqdm(problems, desc="Mode Comparison"):
        question = problem["question"]
        target_answer = problem["answer"]

        for mode in modes:
            start_time = time.time()

            try:
                if mode == "off":
                    # Baseline: LLM only
                    response = llm.generate_with_cot(
                        prompt=question,
                        max_new_tokens=512,
                        temperature=0.7,
                        num_samples=1,
                    )[0]
                    full_text = response["full_text"]
                    jepa_loss = None

                elif mode == "rerank":
                    # Re-ranking mode
                    result = rerank_cots_with_jepa(
                        prompt=question,
                        llm=llm,
                        rjepa_client=rjepa_client,
                        num_samples=4,
                    )
                    full_text = result["best_cot"]["full_text"]
                    jepa_loss = result["best_cot"]["jepa_loss"]

                elif mode == "nudge":
                    # Nudge mode
                    result = nudge_with_regeneration(
                        prompt=question,
                        llm=llm,
                        rjepa_client=rjepa_client,
                        max_attempts=3,
                    )
                    full_text = result["full_text"]
                    jepa_loss = result.get("final_jepa_loss")

                elif mode == "plan":
                    # Plan mode
                    result = auto_complete_missing_steps(
                        prompt=question,
                        llm=llm,
                        rjepa_client=rjepa_client,
                        num_expected_steps=5,
                    )
                    full_text = result["full_text"]
                    jepa_loss = None

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Extract answer
                predicted_answer = extract_answer(full_text, answer_type="numeric")
                is_correct = (predicted_answer == target_answer) if predicted_answer else False

                elapsed = time.time() - start_time

                results_by_mode[mode].append({
                    "problem_id": problem["problem_id"],
                    "question": question,
                    "target": target_answer,
                    "prediction": predicted_answer,
                    "is_correct": is_correct,
                    "jepa_loss": jepa_loss,
                    "elapsed_time": elapsed,
                    "full_response": full_text,
                })

            except Exception as e:
                logger.error(f"Mode {mode} failed for {problem['problem_id']}: {e}")
                results_by_mode[mode].append({
                    "problem_id": problem["problem_id"],
                    "is_correct": False,
                    "error": str(e),
                })

    # Compute summary statistics
    summary = {}
    for mode in modes:
        mode_results = results_by_mode[mode]
        accuracy = sum(r.get("is_correct", False) for r in mode_results) / len(mode_results)

        summary[mode] = {
            "accuracy": accuracy,
            "num_correct": sum(r.get("is_correct", False) for r in mode_results),
            "num_samples": len(mode_results),
        }

        # Add JEPA stats if available
        jepa_losses = [r["jepa_loss"] for r in mode_results if r.get("jepa_loss") is not None]
        if jepa_losses:
            import numpy as np
            summary[mode]["jepa_loss_mean"] = float(np.mean(jepa_losses))
            summary[mode]["jepa_loss_std"] = float(np.std(jepa_losses))

        logger.info(f"Mode {mode}: accuracy={accuracy:.4f}")

    return {
        "summary": summary,
        "results_by_mode": results_by_mode,
        "num_samples": len(problems),
    }
