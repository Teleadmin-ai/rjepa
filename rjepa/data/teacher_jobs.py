"""
Teacher orchestration jobs (Prefect workflows).

Provides high-level flows for dataset generation and validation.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd

from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

from rjepa.config import settings
from rjepa.teacher.client import TeacherClient, MultiSourceTeacher
from rjepa.teacher.generator import ProblemGenerator, CoTGenerator, DatasetGenerator
from rjepa.teacher.validator import Validator
from rjepa.teacher.budget_tracker import BudgetTracker
from rjepa.data.schemas import Problem, ChainOfThought

logger = logging.getLogger(__name__)


@task(
    name="initialize_teacher_clients",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24),
)
def initialize_teacher_clients() -> MultiSourceTeacher:
    """
    Initialize teacher LLM clients from settings.

    Returns:
        MultiSourceTeacher with configured clients
    """
    teacher = MultiSourceTeacher()

    # Add Claude client
    if settings.teacher_claude_api_key:
        claude_client = TeacherClient(
            base_url=settings.teacher_claude_base_url,
            api_key=settings.teacher_claude_api_key,
            model=settings.teacher_claude_model,
        )
        teacher.add_client("claude", claude_client)
        logger.info("Added Claude teacher client")

    # Add GPT client
    if settings.teacher_gpt_api_key:
        gpt_client = TeacherClient(
            base_url=settings.teacher_gpt_base_url,
            api_key=settings.teacher_gpt_api_key,
            model=settings.teacher_gpt_model,
        )
        teacher.add_client("gpt", gpt_client)
        logger.info("Added GPT teacher client")

    return teacher


@task(name="generate_problems")
def generate_problems_task(
    teacher: MultiSourceTeacher,
    domain: str,
    subdomain: Optional[str],
    difficulty: str,
    num_problems: int,
) -> List[Problem]:
    """
    Generate problems task.

    Args:
        teacher: MultiSourceTeacher instance
        domain: Problem domain
        subdomain: Optional subdomain
        difficulty: Difficulty level
        num_problems: Number of problems

    Returns:
        List of Problem objects
    """
    # Use first available client
    client = list(teacher.clients.values())[0]

    generator = ProblemGenerator(client)
    problems = generator.generate_problems(
        domain=domain,
        subdomain=subdomain,
        difficulty=difficulty,
        num_problems=num_problems,
    )

    logger.info(f"Generated {len(problems)} problems")
    return problems


@task(name="generate_cots")
def generate_cots_task(
    teacher: MultiSourceTeacher,
    problems: List[Problem],
    cots_per_problem: int,
) -> List[ChainOfThought]:
    """
    Generate CoT solutions task.

    Args:
        teacher: MultiSourceTeacher instance
        problems: List of problems
        cots_per_problem: CoT samples per problem

    Returns:
        List of ChainOfThought objects
    """
    # Use first available client
    client = list(teacher.clients.values())[0]

    generator = CoTGenerator(client)
    all_cots = []

    for i, problem in enumerate(problems):
        logger.info(f"Generating CoT for problem {i+1}/{len(problems)}")
        cots = generator.generate_cot(problem, num_samples=cots_per_problem)
        all_cots.extend(cots)

    logger.info(f"Generated {len(all_cots)} CoT solutions")
    return all_cots


@task(name="validate_cots")
def validate_cots_task(
    problems: List[Problem],
    cots: List[ChainOfThought],
) -> List[ChainOfThought]:
    """
    Validate CoT solutions task.

    Args:
        problems: List of problems
        cots: List of CoT solutions

    Returns:
        List of validated CoT (is_valid field updated)
    """
    validator = Validator()

    # Match CoT to problems
    problem_map = {p.problem_id: p for p in problems}
    validated_cots = []

    for cot in cots:
        problem = problem_map.get(cot.problem_id)
        if problem:
            is_valid, reason = validator.validate(problem, cot)
            cot.is_valid = is_valid
            cot.validation_reason = reason
            validated_cots.append(cot)

    valid_count = sum(1 for cot in validated_cots if cot.is_valid)
    logger.info(f"Validated {len(validated_cots)} CoT: {valid_count} valid ({valid_count/len(validated_cots)*100:.1f}%)")

    return validated_cots


@task(name="save_dataset")
def save_dataset_task(
    problems: List[Problem],
    cots: List[ChainOfThought],
    output_dir: str,
    split: str = "train",
):
    """
    Save dataset to parquet files.

    Args:
        problems: List of problems
        cots: List of CoT solutions
        output_dir: Output directory
        split: Dataset split ("train", "val", "test")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save problems
    problems_df = pd.DataFrame([p.model_dump() for p in problems])
    problems_path = output_path / f"problems_{split}.parquet"
    problems_df.to_parquet(problems_path)
    logger.info(f"Saved {len(problems)} problems to {problems_path}")

    # Save CoT
    cots_df = pd.DataFrame([c.model_dump() for c in cots])
    cots_path = output_path / f"cots_{split}.parquet"
    cots_df.to_parquet(cots_path)
    logger.info(f"Saved {len(cots)} CoT to {cots_path}")


@flow(name="generate_dataset_flow")
def generate_dataset_flow(
    domain: str,
    num_problems: int = 100,
    cots_per_problem: int = 3,
    difficulty: str = "medium",
    subdomain: Optional[str] = None,
    output_dir: str = "data/processed",
    split: str = "train",
    max_budget_usd: float = 50.0,
):
    """
    Complete dataset generation flow.

    Args:
        domain: Problem domain
        num_problems: Number of problems
        cots_per_problem: CoT samples per problem
        difficulty: Difficulty level
        subdomain: Optional subdomain
        output_dir: Output directory
        split: Dataset split
        max_budget_usd: Maximum budget for API calls

    Returns:
        Dict with stats
    """
    logger.info(f"Starting dataset generation: {domain} x {num_problems}")

    # Initialize budget tracker
    budget_tracker = BudgetTracker(max_budget_usd=max_budget_usd)

    # Initialize teacher clients
    teacher = initialize_teacher_clients()

    # Generate problems
    problems = generate_problems_task(
        teacher=teacher,
        domain=domain,
        subdomain=subdomain,
        difficulty=difficulty,
        num_problems=num_problems,
    )

    # Generate CoT
    cots = generate_cots_task(
        teacher=teacher,
        problems=problems,
        cots_per_problem=cots_per_problem,
    )

    # Validate CoT
    validated_cots = validate_cots_task(problems=problems, cots=cots)

    # Filter valid CoT only
    valid_cots = [cot for cot in validated_cots if cot.is_valid]

    # Save dataset
    save_dataset_task(
        problems=problems,
        cots=valid_cots,
        output_dir=output_dir,
        split=split,
    )

    # Return stats
    stats = {
        "domain": domain,
        "num_problems": len(problems),
        "num_cots_generated": len(cots),
        "num_cots_valid": len(valid_cots),
        "validation_rate": len(valid_cots) / len(cots) if cots else 0.0,
        "budget_used_usd": budget_tracker.get_total_cost(),
        "output_dir": output_dir,
    }

    logger.info(f"Dataset generation complete: {stats}")
    return stats


if __name__ == "__main__":
    # Example usage
    generate_dataset_flow(
        domain="math",
        num_problems=10,
        cots_per_problem=2,
        difficulty="easy",
        subdomain="algebra",
        max_budget_usd=10.0,
    )
