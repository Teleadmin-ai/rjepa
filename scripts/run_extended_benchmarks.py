"""
CLI tool for running comprehensive extended benchmarks evaluation.

Usage:
    # Run all extended benchmarks (quick mode)
    python scripts/run_extended_benchmarks.py \
        --llm qwen3-8b \
        --rjepa-checkpoint data/checkpoints/rjepa-qwen3-8b/latest.pth \
        --mode rerank \
        --quick

    # Run full MMLU (all 57 subjects)
    python scripts/run_extended_benchmarks.py \
        --llm qwen3-8b \
        --rjepa-checkpoint data/checkpoints/rjepa-qwen3-8b/latest.pth \
        --benchmarks mmlu \
        --mmlu-category all \
        --max-samples 1000

    # Run STEM subjects only
    python scripts/run_extended_benchmarks.py \
        --llm qwen3-8b \
        --rjepa-checkpoint data/checkpoints/rjepa-qwen3-8b/latest.pth \
        --benchmarks mmlu \
        --mmlu-category stem
"""
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def evaluate_on_benchmark(
    benchmark_name: str,
    problems: List[Dict[str, Any]],
    llm: LLMAdapter,
    rjepa_client: RJEPAClient,
    mode: str = "rerank",
    num_samples: int = 4,
) -> Dict[str, Any]:
    """
    Evaluate R-JEPA on a benchmark.

    Args:
        benchmark_name: Name of benchmark
        problems: List of problems
        llm: LLM adapter
        rjepa_client: R-JEPA client
        mode: "rerank", "nudge", or "plan"
        num_samples: Number of candidates for rerank

    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating on {benchmark_name}: {len(problems)} problems")

    baseline_correct = 0
    jepa_correct = 0
    results = []

    for i, problem in enumerate(problems):
        try:
            # Baseline (JEPA off)
            baseline_response = llm.generate_with_cot(
                prompt=problem["statement"],
                max_new_tokens=512,
                temperature=0.7,
                num_samples=1,
            )[0]

            baseline_answer = extract_answer(baseline_response["full_text"])
            baseline_is_correct = check_answer(
                baseline_answer, problem["answer_gold"], problem["domain"]
            )

            if baseline_is_correct:
                baseline_correct += 1

            # JEPA (rerank mode)
            if mode == "rerank":
                jepa_result = rerank_cots_with_jepa(
                    prompt=problem["statement"],
                    llm=llm,
                    rjepa_client=rjepa_client,
                    num_samples=num_samples,
                )
                jepa_answer = extract_answer(jepa_result["best_cot"]["full_text"])
                jepa_loss = jepa_result["best_cot"]["jepa_loss"]
            else:
                # TODO: implement nudge/plan modes for extended benchmarks
                jepa_answer = baseline_answer
                jepa_loss = None

            jepa_is_correct = check_answer(
                jepa_answer, problem["answer_gold"], problem["domain"]
            )

            if jepa_is_correct:
                jepa_correct += 1

            results.append(
                {
                    "problem_id": problem["problem_id"],
                    "domain": problem["domain"],
                    "subdomain": problem["subdomain"],
                    "baseline_correct": baseline_is_correct,
                    "jepa_correct": jepa_is_correct,
                    "jepa_loss": jepa_loss,
                }
            )

            if (i + 1) % 10 == 0:
                logger.info(
                    f"  Progress: {i + 1}/{len(problems)} "
                    f"(Baseline: {baseline_correct}/{i+1}, "
                    f"JEPA: {jepa_correct}/{i+1})"
                )

        except Exception as e:
            logger.error(f"Error evaluating problem {problem['problem_id']}: {e}")
            results.append(
                {
                    "problem_id": problem["problem_id"],
                    "baseline_correct": False,
                    "jepa_correct": False,
                    "error": str(e),
                }
            )

    # Compute final metrics
    baseline_accuracy = baseline_correct / len(problems)
    jepa_accuracy = jepa_correct / len(problems)
    delta = jepa_accuracy - baseline_accuracy

    logger.info(f"\n{benchmark_name} Results:")
    logger.info(f"  Baseline accuracy: {baseline_accuracy:.4f}")
    logger.info(f"  JEPA accuracy: {jepa_accuracy:.4f}")
    logger.info(f"  Delta: {delta:+.4f} ({delta/baseline_accuracy*100:+.2f}%)")

    return {
        "benchmark": benchmark_name,
        "num_problems": len(problems),
        "baseline_accuracy": baseline_accuracy,
        "jepa_accuracy": jepa_accuracy,
        "delta": delta,
        "relative_improvement": (delta / baseline_accuracy * 100)
        if baseline_accuracy > 0
        else 0,
        "results": results,
    }


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer from generated text.

    For multiple-choice (MMLU, ARC, HellaSwag): extract letter (A, B, C, D)
    For open-ended: extract last sentence or numeric value
    """
    import re

    # Try to find "Answer: X" pattern
    match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try to find standalone letter at end
    match = re.search(r"\b([A-D])\b(?!.*\b[A-D]\b)", text)
    if match:
        return match.group(1).upper()

    # Fallback: return last non-empty line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return lines[-1]

    return None


def check_answer(
    predicted: Optional[str], target: Optional[str], domain: str
) -> bool:
    """
    Check if predicted answer matches target.

    Args:
        predicted: Predicted answer
        target: Gold answer
        domain: Problem domain

    Returns:
        True if correct
    """
    if predicted is None or target is None:
        return False

    # Normalize
    predicted = predicted.strip().upper()
    target = target.strip().upper()

    # Exact match
    if predicted == target:
        return True

    # For multiple-choice: check if letter is in predicted string
    if domain in ["general_knowledge", "science", "commonsense"]:
        return target in predicted

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run extended benchmarks evaluation (MMLU, BBH, ARC, HellaSwag)"
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="qwen3-8b",
        help="LLM tag (e.g., qwen3-8b, llama3-70b)",
    )
    parser.add_argument(
        "--rjepa-checkpoint",
        type=str,
        required=True,
        help="Path to R-JEPA checkpoint",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rerank",
        choices=["rerank", "nudge", "plan"],
        help="R-JEPA mode",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["mmlu", "bbh", "arc", "hellaswag"],
        choices=["mmlu", "bbh", "arc", "hellaswag"],
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--mmlu-category",
        type=str,
        default="stem",
        choices=["stem", "humanities", "social_sciences", "other", "all"],
        help="MMLU category",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max samples per benchmark/subject",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (50 samples per benchmark)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/extended_benchmarks",
        help="Output directory",
    )

    args = parser.parse_args()

    if args.quick:
        args.max_samples = 50
        logger.info("Quick mode: max 50 samples per benchmark")

    logger.info("=" * 80)
    logger.info("EXTENDED BENCHMARKS EVALUATION")
    logger.info("=" * 80)
    logger.info(f"LLM: {args.llm}")
    logger.info(f"R-JEPA checkpoint: {args.rjepa_checkpoint}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Benchmarks: {', '.join(args.benchmarks)}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info("=" * 80)

    # Initialize LLM and R-JEPA client
    logger.info(f"\nLoading LLM: {args.llm}...")
    llm = LLMAdapter(model_name=f"Qwen/Qwen3-8B-Instruct", layer_to_extract=-2)

    logger.info(f"Connecting to R-JEPA service...")
    rjepa_client = RJEPAClient(base_url="http://localhost:8100")

    # Load benchmarks
    benchmark_suite = {}

    if "mmlu" in args.benchmarks:
        logger.info(f"\nLoading MMLU ({args.mmlu_category})...")
        benchmark_suite["mmlu"] = load_mmlu(
            category=args.mmlu_category,
            max_samples_per_subject=args.max_samples,
        )

    if "bbh" in args.benchmarks:
        logger.info(f"\nLoading Big-Bench Hard...")
        benchmark_suite["bbh"] = load_bbh(max_samples_per_task=args.max_samples)

    if "arc" in args.benchmarks:
        logger.info(f"\nLoading ARC-Challenge...")
        benchmark_suite["arc"] = load_arc(
            challenge_only=True, max_samples=args.max_samples
        )

    if "hellaswag" in args.benchmarks:
        logger.info(f"\nLoading HellaSwag...")
        benchmark_suite["hellaswag"] = load_hellaswag(max_samples=args.max_samples)

    logger.info(
        f"\nLoaded {sum(len(problems) for problems in benchmark_suite.values())} total problems"
    )

    # Evaluate on each benchmark
    all_results = {}

    for benchmark_name, problems in benchmark_suite.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating: {benchmark_name.upper()}")
        logger.info(f"{'=' * 80}")

        # Convert Problem objects to dicts
        if problems and hasattr(problems[0], "model_dump"):
            problems = [p.model_dump() for p in problems]

        results = evaluate_on_benchmark(
            benchmark_name=benchmark_name,
            problems=problems,
            llm=llm,
            rjepa_client=rjepa_client,
            mode=args.mode,
        )

        all_results[benchmark_name] = results

    # Compute aggregate metrics
    total_problems = sum(r["num_problems"] for r in all_results.values())
    weighted_baseline = sum(
        r["baseline_accuracy"] * r["num_problems"] for r in all_results.values()
    ) / total_problems
    weighted_jepa = sum(
        r["jepa_accuracy"] * r["num_problems"] for r in all_results.values()
    ) / total_problems
    weighted_delta = weighted_jepa - weighted_baseline

    logger.info(f"\n{'=' * 80}")
    logger.info("AGGREGATE RESULTS (ACROSS ALL BENCHMARKS)")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total problems: {total_problems}")
    logger.info(f"Weighted baseline accuracy: {weighted_baseline:.4f}")
    logger.info(f"Weighted JEPA accuracy: {weighted_jepa:.4f}")
    logger.info(
        f"Weighted delta: {weighted_delta:+.4f} ({weighted_delta/weighted_baseline*100:+.2f}%)"
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"extended_benchmarks_{timestamp}.json"

    full_results = {
        "llm": args.llm,
        "rjepa_checkpoint": args.rjepa_checkpoint,
        "mode": args.mode,
        "benchmarks": args.benchmarks,
        "mmlu_category": args.mmlu_category,
        "max_samples": args.max_samples,
        "timestamp": timestamp,
        "aggregate": {
            "total_problems": total_problems,
            "weighted_baseline_accuracy": weighted_baseline,
            "weighted_jepa_accuracy": weighted_jepa,
            "weighted_delta": weighted_delta,
            "relative_improvement": (weighted_delta / weighted_baseline * 100)
            if weighted_baseline > 0
            else 0,
        },
        "per_benchmark": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
