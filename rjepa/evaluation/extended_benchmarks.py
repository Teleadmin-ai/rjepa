"""
Extended Benchmarks: MMLU, Big-Bench Hard, and more.

These benchmarks test broader capabilities beyond math/code:
- MMLU: Multitask Language Understanding (57 subjects, general knowledge)
- Big-Bench Hard: Complex reasoning tasks (BBH subset)
- Additional: ARC, HellaSwag, TruthfulQA, SIQA
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from datasets import load_dataset

from rjepa.data.schemas import Problem

logger = logging.getLogger(__name__)


# MMLU subject categories
MMLU_SUBJECTS = {
    "stem": [
        "abstract_algebra",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "social_sciences": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "other": [
        "anatomy",
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    ],
}


def load_mmlu(
    subjects: Optional[List[str]] = None,
    split: str = "test",
    max_samples_per_subject: Optional[int] = None,
    category: Optional[str] = None,
) -> List[Problem]:
    """
    Load MMLU (Massive Multitask Language Understanding) benchmark.

    57 subjects across STEM, humanities, social sciences, and other domains.
    Each problem is multiple-choice (4 options: A, B, C, D).

    Args:
        subjects: List of subject names (e.g., ["high_school_mathematics"])
                  If None, load all subjects
        split: Dataset split ("test", "validation", "dev")
        max_samples_per_subject: Max samples per subject (for quick testing)
        category: Load entire category ("stem", "humanities", "social_sciences", "other")

    Returns:
        List of Problem objects
    """
    logger.info(f"Loading MMLU benchmark (split={split})...")

    # Determine subjects to load
    if category:
        if category not in MMLU_SUBJECTS:
            raise ValueError(
                f"Unknown MMLU category: {category}. "
                f"Valid: {list(MMLU_SUBJECTS.keys())}"
            )
        subjects = MMLU_SUBJECTS[category]
        logger.info(f"Loading MMLU category '{category}': {len(subjects)} subjects")
    elif subjects is None:
        # Load all subjects
        subjects = []
        for subj_list in MMLU_SUBJECTS.values():
            subjects.extend(subj_list)
        logger.info(f"Loading all MMLU subjects: {len(subjects)} total")

    problems = []

    for subject in subjects:
        try:
            # Load from HuggingFace datasets
            dataset = load_dataset("cais/mmlu", subject, split=split)

            if max_samples_per_subject:
                dataset = dataset.select(range(min(len(dataset), max_samples_per_subject)))

            for idx, sample in enumerate(dataset):
                # MMLU format:
                # - question: str
                # - choices: List[str] (4 options)
                # - answer: int (0-3, corresponds to A-D)

                question = sample["question"]
                choices = sample["choices"]
                answer_idx = sample["answer"]
                answer_letter = chr(ord("A") + answer_idx)  # 0->A, 1->B, etc.

                # Format statement with choices
                choices_str = "\n".join(
                    f"{chr(ord('A')+i)}. {choice}" for i, choice in enumerate(choices)
                )

                statement = f"{question}\n\n{choices_str}\n\nAnswer:"

                problem = Problem(
                    problem_id=f"mmlu_{subject}_{idx}",
                    domain="general_knowledge",
                    subdomain=subject,
                    source=f"mmlu_{subject}",
                    difficulty="varies",  # MMLU doesn't specify difficulty
                    statement=statement,
                    answer_gold=answer_letter,  # "A", "B", "C", or "D"
                    meta_course={"category": _get_mmlu_category(subject)},
                )

                problems.append(problem)

            logger.debug(f"  Loaded {len(dataset)} samples from {subject}")

        except Exception as e:
            logger.warning(f"Failed to load MMLU subject '{subject}': {e}")
            continue

    logger.info(f"Loaded {len(problems)} problems from MMLU ({len(subjects)} subjects)")

    return problems


def _get_mmlu_category(subject: str) -> str:
    """Get MMLU category for a subject."""
    for category, subjects in MMLU_SUBJECTS.items():
        if subject in subjects:
            return category
    return "other"


def load_bbh(
    tasks: Optional[List[str]] = None,
    max_samples_per_task: Optional[int] = None,
) -> List[Problem]:
    """
    Load Big-Bench Hard (BBH) benchmark.

    BBH is a subset of 23 challenging tasks from Big-Bench that are hard for LLMs.
    Focuses on complex reasoning (logical deduction, tracking objects, etc.).

    Args:
        tasks: List of task names (e.g., ["logical_deduction", "tracking_shuffled_objects"])
               If None, load all 23 tasks
        max_samples_per_task: Max samples per task

    Returns:
        List of Problem objects
    """
    logger.info("Loading Big-Bench Hard (BBH) benchmark...")

    # BBH task list (23 tasks)
    BBH_TASKS = [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ]

    if tasks is None:
        tasks = BBH_TASKS
        logger.info(f"Loading all BBH tasks: {len(tasks)} total")

    problems = []

    for task in tasks:
        try:
            # Load from HuggingFace datasets
            dataset = load_dataset("lukaemon/bbh", task, split="test")

            if max_samples_per_task:
                dataset = dataset.select(range(min(len(dataset), max_samples_per_task)))

            for idx, sample in enumerate(dataset):
                # BBH format:
                # - input: str (problem statement)
                # - target: str (correct answer)

                statement = sample["input"]
                answer = sample["target"]

                problem = Problem(
                    problem_id=f"bbh_{task}_{idx}",
                    domain="complex_reasoning",
                    subdomain=task,
                    source=f"bbh_{task}",
                    difficulty="hard",  # BBH is by definition hard
                    statement=statement,
                    answer_gold=answer,
                )

                problems.append(problem)

            logger.debug(f"  Loaded {len(dataset)} samples from {task}")

        except Exception as e:
            logger.warning(f"Failed to load BBH task '{task}': {e}")
            continue

    logger.info(f"Loaded {len(problems)} problems from BBH ({len(tasks)} tasks)")

    return problems


def load_arc(
    challenge_only: bool = True,
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[Problem]:
    """
    Load ARC (AI2 Reasoning Challenge) benchmark.

    ARC-Challenge: 1,172 harder questions (grade-school science)
    ARC-Easy: 2,376 easier questions

    Args:
        challenge_only: Load only ARC-Challenge (harder subset)
        split: Dataset split ("test", "validation")
        max_samples: Max samples to load

    Returns:
        List of Problem objects
    """
    subset = "ARC-Challenge" if challenge_only else "ARC-Easy"
    logger.info(f"Loading ARC benchmark ({subset}, split={split})...")

    try:
        dataset = load_dataset("allenai/ai2_arc", subset, split=split)

        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))

        problems = []

        for idx, sample in enumerate(dataset):
            # ARC format:
            # - question: str
            # - choices: {"text": List[str], "label": List[str]}
            # - answerKey: str ("A", "B", "C", "D", or "1", "2", "3", "4")

            question = sample["question"]
            choices = sample["choices"]
            answer_key = sample["answerKey"]

            # Format choices
            choices_str = "\n".join(
                f"{label}. {text}"
                for label, text in zip(choices["label"], choices["text"])
            )

            statement = f"{question}\n\n{choices_str}\n\nAnswer:"

            problem = Problem(
                problem_id=f"arc_{subset.lower()}_{idx}",
                domain="science",
                subdomain="grade_school_science",
                source=f"arc_{subset.lower()}",
                difficulty="hard" if challenge_only else "easy",
                statement=statement,
                answer_gold=answer_key,
            )

            problems.append(problem)

        logger.info(f"Loaded {len(problems)} problems from ARC ({subset})")

        return problems

    except Exception as e:
        logger.error(f"Failed to load ARC: {e}")
        return []


def load_hellaswag(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> List[Problem]:
    """
    Load HellaSwag benchmark (commonsense reasoning, sentence completion).

    Args:
        split: Dataset split ("validation", "train")
        max_samples: Max samples to load

    Returns:
        List of Problem objects
    """
    logger.info(f"Loading HellaSwag benchmark (split={split})...")

    try:
        dataset = load_dataset("Rowan/hellaswag", split=split)

        if max_samples:
            dataset = dataset.select(range(min(len(dataset), max_samples)))

        problems = []

        for idx, sample in enumerate(dataset):
            # HellaSwag format:
            # - ctx: str (context)
            # - endings: List[str] (4 possible endings)
            # - label: int (0-3, correct ending)

            context = sample["ctx"]
            endings = sample["endings"]
            label = int(sample["label"])
            answer_letter = chr(ord("A") + label)

            # Format endings
            endings_str = "\n".join(
                f"{chr(ord('A')+i)}. {ending}" for i, ending in enumerate(endings)
            )

            statement = (
                f"Complete the sentence:\n\n{context}\n\n{endings_str}\n\nAnswer:"
            )

            problem = Problem(
                problem_id=f"hellaswag_{idx}",
                domain="commonsense",
                subdomain="sentence_completion",
                source="hellaswag",
                difficulty="medium",
                statement=statement,
                answer_gold=answer_letter,
            )

            problems.append(problem)

        logger.info(f"Loaded {len(problems)} problems from HellaSwag")

        return problems

    except Exception as e:
        logger.error(f"Failed to load HellaSwag: {e}")
        return []


def create_extended_benchmark_suite(
    include_mmlu: bool = True,
    include_bbh: bool = True,
    include_arc: bool = True,
    include_hellaswag: bool = False,  # Very large, optional
    mmlu_category: Optional[str] = "stem",  # "stem", "humanities", etc., or None for all
    max_samples_per_benchmark: Optional[int] = 100,  # For quick testing
) -> Dict[str, List[Problem]]:
    """
    Create a comprehensive benchmark suite combining multiple benchmarks.

    Args:
        include_mmlu: Include MMLU (general knowledge)
        include_bbh: Include Big-Bench Hard (complex reasoning)
        include_arc: Include ARC (science reasoning)
        include_hellaswag: Include HellaSwag (commonsense)
        mmlu_category: MMLU category to load ("stem", "all", etc.)
        max_samples_per_benchmark: Max samples per benchmark (None = all)

    Returns:
        Dict mapping benchmark name to list of problems
    """
    logger.info("Creating extended benchmark suite...")

    suite = {}

    if include_mmlu:
        if mmlu_category == "all":
            mmlu_problems = load_mmlu(max_samples_per_subject=max_samples_per_benchmark)
        elif mmlu_category:
            mmlu_problems = load_mmlu(
                category=mmlu_category, max_samples_per_subject=max_samples_per_benchmark
            )
        else:
            mmlu_problems = load_mmlu(
                subjects=["high_school_mathematics", "high_school_physics"],
                max_samples_per_subject=max_samples_per_benchmark,
            )
        suite["mmlu"] = mmlu_problems

    if include_bbh:
        bbh_problems = load_bbh(max_samples_per_task=max_samples_per_benchmark)
        suite["bbh"] = bbh_problems

    if include_arc:
        arc_problems = load_arc(
            challenge_only=True, max_samples=max_samples_per_benchmark
        )
        suite["arc"] = arc_problems

    if include_hellaswag:
        hellaswag_problems = load_hellaswag(max_samples=max_samples_per_benchmark)
        suite["hellaswag"] = hellaswag_problems

    total_problems = sum(len(problems) for problems in suite.values())
    logger.info(
        f"Extended benchmark suite created: {len(suite)} benchmarks, "
        f"{total_problems} total problems"
    )

    return suite
