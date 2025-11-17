"""
Main evaluation pipeline for R-JEPA.

Orchestrates benchmarking, A/B testing, and correlation analysis.
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime

# Prefect for orchestration
from prefect import flow, task

# R-JEPA imports
from rjepa.evaluation import (
    load_gsm8k,
    load_math,
    load_humaneval,
    compute_metrics_summary,
    run_ab_test,
    compare_modes,
)
from rjepa.evaluation.extended_benchmarks import (
    load_mmlu,
    load_bbh,
    load_arc,
    load_hellaswag,
    create_extended_benchmark_suite,
)
from rjepa.llm.adapter import LLMAdapter
from rjepa.jepa.client import RJEPAClient
from rjepa.inference import rerank_cots_with_jepa

logger = logging.getLogger(__name__)


@task(name="load_benchmark")
def load_benchmark_task(
    benchmark_name: str,
    split: str = "test",
    num_samples: Optional[int] = None,
    difficulty: Optional[str] = None,
    category: Optional[str] = None,  # For MMLU
) -> List[Dict[str, Any]]:
    """
    Load benchmark dataset.

    Args:
        benchmark_name: "gsm8k", "math", "humaneval", "mmlu", "bbh", "arc", "hellaswag"
        split: "train" or "test"
        num_samples: Limit number of samples
        difficulty: Filter by difficulty (MATH only)
        category: MMLU category ("stem", "humanities", "social_sciences", "other", "all")

    Returns:
        List of problems
    """
    logger.info(f"Loading benchmark: {benchmark_name} ({split} split, {num_samples or 'all'} samples)")

    if benchmark_name == "gsm8k":
        problems = load_gsm8k(split=split, num_samples=num_samples)
    elif benchmark_name == "math":
        problems = load_math(split=split, num_samples=num_samples, difficulty=difficulty)
    elif benchmark_name == "humaneval":
        problems = load_humaneval(num_samples=num_samples)

    # Extended benchmarks (Phase 17)
    elif benchmark_name == "mmlu":
        problems = load_mmlu(
            category=category,
            split=split,
            max_samples_per_subject=num_samples,
        )
    elif benchmark_name == "bbh":
        problems = load_bbh(max_samples_per_task=num_samples)
    elif benchmark_name == "arc":
        problems = load_arc(
            challenge_only=True,
            split=split,
            max_samples=num_samples,
        )
    elif benchmark_name == "hellaswag":
        problems = load_hellaswag(split=split, max_samples=num_samples)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    # Convert Problem objects to dicts for compatibility
    if problems and hasattr(problems[0], 'model_dump'):
        problems = [p.model_dump() for p in problems]

    return problems


@task(name="evaluate_baseline")
def evaluate_baseline_task(
    problems: List[Dict[str, Any]],
    llm: LLMAdapter,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Evaluate baseline (JEPA off) on problems.

    Args:
        problems: List of problems
        llm: LLM adapter
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        List of results with predictions and correctness
    """
    from rjepa.evaluation.metrics import extract_answer

    logger.info(f"Evaluating baseline on {len(problems)} problems...")

    results = []
    for i, problem in enumerate(problems):
        try:
            # Generate answer
            response = llm.generate_with_cot(
                prompt=problem["question"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_samples=1,
            )[0]

            # Extract answer
            predicted_answer = extract_answer(response["full_text"], answer_type="numeric")
            is_correct = (predicted_answer == problem["answer"]) if predicted_answer else False

            results.append({
                "problem_id": problem["problem_id"],
                "question": problem["question"],
                "target": problem["answer"],
                "prediction": predicted_answer,
                "is_correct": is_correct,
                "full_response": response["full_text"],
            })

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(problems)} problems")

        except Exception as e:
            logger.error(f"Error evaluating problem {problem['problem_id']}: {e}")
            results.append({
                "problem_id": problem["problem_id"],
                "is_correct": False,
                "error": str(e),
            })

    accuracy = sum(r.get("is_correct", False) for r in results) / len(results)
    logger.info(f"Baseline accuracy: {accuracy:.4f}")

    return results


@task(name="evaluate_with_jepa")
def evaluate_with_jepa_task(
    problems: List[Dict[str, Any]],
    llm: LLMAdapter,
    rjepa_client: RJEPAClient,
    mode: str = "rerank",
    num_samples: int = 4,
) -> List[Dict[str, Any]]:
    """
    Evaluate with R-JEPA on problems.

    Args:
        problems: List of problems
        llm: LLM adapter
        rjepa_client: R-JEPA client
        mode: "rerank", "nudge", or "plan"
        num_samples: Number of candidates for rerank

    Returns:
        List of results with JEPA scores
    """
    from rjepa.evaluation.metrics import extract_answer
    from rjepa.inference import nudge_with_regeneration, auto_complete_missing_steps

    logger.info(f"Evaluating with R-JEPA (mode={mode}) on {len(problems)} problems...")

    results = []
    for i, problem in enumerate(problems):
        try:
            if mode == "rerank":
                result = rerank_cots_with_jepa(
                    prompt=problem["question"],
                    llm=llm,
                    rjepa_client=rjepa_client,
                    num_samples=num_samples,
                )
                full_text = result["best_cot"]["full_text"]
                jepa_loss = result["best_cot"]["jepa_loss"]

            elif mode == "nudge":
                result = nudge_with_regeneration(
                    prompt=problem["question"],
                    llm=llm,
                    rjepa_client=rjepa_client,
                    max_attempts=3,
                )
                full_text = result["full_text"]
                jepa_loss = result.get("final_jepa_loss")

            elif mode == "plan":
                result = auto_complete_missing_steps(
                    prompt=problem["question"],
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
            is_correct = (predicted_answer == problem["answer"]) if predicted_answer else False

            results.append({
                "problem_id": problem["problem_id"],
                "question": problem["question"],
                "target": problem["answer"],
                "prediction": predicted_answer,
                "is_correct": is_correct,
                "jepa_loss": jepa_loss,
                "full_response": full_text,
            })

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(problems)} problems")

        except Exception as e:
            logger.error(f"Error evaluating problem {problem['problem_id']}: {e}")
            results.append({
                "problem_id": problem["problem_id"],
                "is_correct": False,
                "error": str(e),
            })

    accuracy = sum(r.get("is_correct", False) for r in results) / len(results)
    logger.info(f"R-JEPA ({mode}) accuracy: {accuracy:.4f}")

    return results


@task(name="compute_metrics")
def compute_metrics_task(
    baseline_results: List[Dict[str, Any]],
    jepa_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics comparing baseline vs JEPA.

    Args:
        baseline_results: Results from baseline evaluation
        jepa_results: Results from JEPA evaluation

    Returns:
        Dictionary of metrics
    """
    logger.info("Computing metrics...")

    # Compute individual metrics
    baseline_metrics = compute_metrics_summary(baseline_results, include_correlation=False)
    jepa_metrics = compute_metrics_summary(jepa_results, include_correlation=True)

    # Compute delta
    delta_accuracy = jepa_metrics["accuracy"] - baseline_metrics["accuracy"]
    relative_improvement = (delta_accuracy / baseline_metrics["accuracy"] * 100) if baseline_metrics["accuracy"] > 0 else 0

    metrics = {
        "baseline": baseline_metrics,
        "jepa": jepa_metrics,
        "delta_accuracy": delta_accuracy,
        "relative_improvement": relative_improvement,
        "num_samples": len(baseline_results),
    }

    logger.info(f"Baseline accuracy: {baseline_metrics['accuracy']:.4f}")
    logger.info(f"JEPA accuracy: {jepa_metrics['accuracy']:.4f}")
    logger.info(f"Delta: {delta_accuracy:+.4f} ({relative_improvement:+.2f}%)")

    return metrics


@task(name="save_results")
def save_results_task(
    results: Dict[str, Any],
    output_path: Path,
):
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


@flow(name="evaluate_rjepa")
def evaluate_rjepa_flow(
    benchmark_name: str,
    llm_name: str,
    rjepa_checkpoint: str,
    mode: str = "rerank",
    split: str = "test",
    num_samples: Optional[int] = None,
    output_dir: str = "./results",
):
    """
    Main evaluation flow for R-JEPA.

    Args:
        benchmark_name: "gsm8k", "math", or "humaneval"
        llm_name: LLM model name (e.g., "Qwen/Qwen3-8B-Instruct")
        rjepa_checkpoint: Path to R-JEPA checkpoint
        mode: "rerank", "nudge", or "plan"
        split: "train" or "test"
        num_samples: Limit number of samples
        output_dir: Directory to save results
    """
    logger.info(f"Starting evaluation: {benchmark_name} with {llm_name} (mode={mode})")

    # Load benchmark
    problems = load_benchmark_task(benchmark_name, split=split, num_samples=num_samples)

    # Initialize LLM and R-JEPA client
    logger.info(f"Loading LLM: {llm_name}")
    llm = LLMAdapter(model_name=llm_name, layer_to_extract=-2)

    logger.info(f"Loading R-JEPA checkpoint: {rjepa_checkpoint}")
    rjepa_client = RJEPAClient(base_url="http://localhost:8100")  # Assume service running

    # Evaluate baseline
    baseline_results = evaluate_baseline_task(problems, llm)

    # Evaluate with JEPA
    jepa_results = evaluate_with_jepa_task(problems, llm, rjepa_client, mode=mode)

    # Compute metrics
    metrics = compute_metrics_task(baseline_results, jepa_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{benchmark_name}_{mode}_{timestamp}.json"

    full_results = {
        "benchmark": benchmark_name,
        "llm": llm_name,
        "rjepa_checkpoint": rjepa_checkpoint,
        "mode": mode,
        "split": split,
        "num_samples": len(problems),
        "timestamp": timestamp,
        "metrics": metrics,
        "baseline_results": baseline_results,
        "jepa_results": jepa_results,
    }

    save_results_task(full_results, output_path)

    logger.info("Evaluation completed!")
    return metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate R-JEPA on benchmarks")

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["gsm8k", "math", "humaneval", "mmlu", "bbh", "arc", "hellaswag"],
        help="Benchmark to evaluate on"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["stem", "humanities", "social_sciences", "other", "all"],
        help="MMLU category (only for --benchmark mmlu)"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="Qwen/Qwen3-8B-Instruct",
        help="LLM model name"
    )
    parser.add_argument(
        "--rjepa-checkpoint",
        type=str,
        required=True,
        help="Path to R-JEPA checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rerank",
        choices=["rerank", "nudge", "plan"],
        help="R-JEPA mode"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples (None = all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run evaluation flow
    evaluate_rjepa_flow(
        benchmark_name=args.benchmark,
        llm_name=args.llm,
        rjepa_checkpoint=args.rjepa_checkpoint,
        mode=args.mode,
        split=args.split,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
