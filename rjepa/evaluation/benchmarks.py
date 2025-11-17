"""
Benchmark dataset loaders for R-JEPA evaluation.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def load_gsm8k(
    split: str = "test",
    num_samples: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load GSM8K (Grade School Math 8K) benchmark.

    Args:
        split: Dataset split ("train" or "test")
        num_samples: Number of samples to load (None = all)
        data_dir: Directory containing GSM8K data

    Returns:
        List of problems, each with:
            - problem_id: str
            - question: str
            - answer: str (numeric)
            - domain: "math"
            - subdomain: "arithmetic"
    """
    logger.info(f"Loading GSM8K ({split} split, {num_samples or 'all'} samples)...")

    # Try to load from HuggingFace datasets
    try:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main", split=split)

        problems = []
        for idx, item in enumerate(dataset):
            if num_samples and idx >= num_samples:
                break

            # Extract final answer from answer field
            # GSM8K answers are formatted as "#### 42"
            answer_text = item["answer"]
            answer_match = answer_text.split("####")[-1].strip()

            problems.append({
                "problem_id": f"gsm8k_{split}_{idx}",
                "question": item["question"],
                "answer": answer_match,
                "domain": "math",
                "subdomain": "arithmetic",
                "difficulty": "easy",
            })

        logger.info(f"Loaded {len(problems)} GSM8K problems")
        return problems

    except ImportError:
        logger.warning("HuggingFace datasets not available, trying local files...")

    # Fallback: load from local JSON file
    if data_dir is None:
        data_dir = Path("data/benchmarks/gsm8k")

    json_file = data_dir / f"{split}.jsonl"

    if not json_file.exists():
        raise FileNotFoundError(
            f"GSM8K data not found at {json_file}. "
            f"Install datasets: pip install datasets"
        )

    problems = []
    with open(json_file) as f:
        for idx, line in enumerate(f):
            if num_samples and idx >= num_samples:
                break

            item = json.loads(line)
            answer_text = item["answer"]
            answer_match = answer_text.split("####")[-1].strip()

            problems.append({
                "problem_id": f"gsm8k_{split}_{idx}",
                "question": item["question"],
                "answer": answer_match,
                "domain": "math",
                "subdomain": "arithmetic",
                "difficulty": "easy",
            })

    logger.info(f"Loaded {len(problems)} GSM8K problems from local file")
    return problems


def load_math(
    split: str = "test",
    num_samples: Optional[int] = None,
    difficulty: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load MATH (Mathematics Aptitude Test of Heuristics) benchmark.

    Args:
        split: Dataset split ("train" or "test")
        num_samples: Number of samples to load (None = all)
        difficulty: Filter by difficulty (1-5, None = all)
        data_dir: Directory containing MATH data

    Returns:
        List of problems
    """
    logger.info(f"Loading MATH ({split} split, difficulty={difficulty or 'all'})...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("hendrycks/math", split=split)

        problems = []
        for idx, item in enumerate(dataset):
            if num_samples and len(problems) >= num_samples:
                break

            # Filter by difficulty if specified
            item_difficulty = item.get("level", "")
            if difficulty and str(difficulty) not in item_difficulty:
                continue

            problems.append({
                "problem_id": f"math_{split}_{idx}",
                "question": item["problem"],
                "answer": item["solution"].split("####")[-1].strip() if "####" in item["solution"] else item["solution"],
                "domain": "math",
                "subdomain": item.get("type", "unknown"),
                "difficulty": item_difficulty,
            })

        logger.info(f"Loaded {len(problems)} MATH problems")
        return problems

    except ImportError:
        logger.warning("HuggingFace datasets not available")
        return []


def load_humaneval(
    num_samples: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load HumanEval (code generation) benchmark.

    Args:
        num_samples: Number of samples to load (None = all)
        data_dir: Directory containing HumanEval data

    Returns:
        List of problems, each with:
            - problem_id: str
            - prompt: str (function signature + docstring)
            - canonical_solution: str
            - test: str (test cases)
            - entry_point: str (function name)
    """
    logger.info(f"Loading HumanEval ({num_samples or 'all'} samples)...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("openai_humaneval", split="test")

        problems = []
        for idx, item in enumerate(dataset):
            if num_samples and idx >= num_samples:
                break

            problems.append({
                "problem_id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
                "domain": "code",
                "subdomain": "python",
                "difficulty": "medium",
            })

        logger.info(f"Loaded {len(problems)} HumanEval problems")
        return problems

    except ImportError:
        logger.warning("HuggingFace datasets not available")
        return []


def create_mini_benchmark(
    benchmark_name: str,
    num_samples: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Create a mini benchmark for quick testing.

    Args:
        benchmark_name: Name of benchmark ("gsm8k", "math", "humaneval")
        num_samples: Number of samples
        seed: Random seed for sampling

    Returns:
        List of sampled problems
    """
    import random
    random.seed(seed)

    if benchmark_name == "gsm8k":
        full_dataset = load_gsm8k(split="test")
    elif benchmark_name == "math":
        full_dataset = load_math(split="test")
    elif benchmark_name == "humaneval":
        full_dataset = load_humaneval()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    if len(full_dataset) <= num_samples:
        return full_dataset

    # Random sample
    sampled = random.sample(full_dataset, num_samples)
    logger.info(f"Sampled {num_samples} from {len(full_dataset)} {benchmark_name} problems")

    return sampled
