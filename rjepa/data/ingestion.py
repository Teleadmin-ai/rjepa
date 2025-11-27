"""
Dataset ingestion from external sources.

Supports:
- HuggingFace datasets (GSM8K, MATH, HumanEval, etc.)
- Custom CSV/JSON files
- User interaction logs
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from dataclasses import asdict

from rjepa.data.schemas import Problem, ChainOfThought

logger = logging.getLogger(__name__)


class HuggingFaceDatasetIngestion:
    """Import datasets from HuggingFace."""

    @staticmethod
    def ingest_gsm8k(
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[Problem]:
        """
        Ingest GSM8K dataset (grade school math).

        Args:
            split: "train" or "test"
            max_samples: Optional limit on number of samples

        Returns:
            List of Problem objects
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package required. Install with: pip install datasets"
            )

        logger.info(f"Loading GSM8K dataset (split={split})...")
        dataset = load_dataset("gsm8k", "main", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        problems = []
        for idx, example in enumerate(dataset):
            # GSM8K format: {"question": str, "answer": str}
            # Answer format: "step1\n####\nfinal_answer"
            answer_parts = example["answer"].split("####")
            answer_gold = answer_parts[-1].strip() if len(answer_parts) > 1 else None

            problem = Problem(
                problem_id=f"gsm8k_{split}_{idx:05d}",
                domain="math",
                subdomain="arithmetic",
                source="gsm8k",
                difficulty="easy",
                statement=example["question"],
                answer_gold=answer_gold,
                meta_course={"dataset": "gsm8k", "split": split},
            )
            problems.append(problem)

        logger.info(f"Ingested {len(problems)} problems from GSM8K ({split})")
        return problems

    @staticmethod
    def ingest_math_dataset(
        split: str = "train",
        max_samples: Optional[int] = None,
        difficulty_filter: Optional[str] = None,
    ) -> List[Problem]:
        """
        Ingest MATH dataset (competition mathematics).

        Args:
            split: "train" or "test"
            max_samples: Optional limit
            difficulty_filter: Optional difficulty level (1-5)

        Returns:
            List of Problem objects
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package required")

        logger.info(f"Loading MATH dataset (split={split})...")
        dataset = load_dataset("hendrycks/math", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        problems = []
        difficulty_map = {1: "easy", 2: "easy", 3: "medium", 4: "hard", 5: "hard"}

        for idx, example in enumerate(dataset):
            # MATH format: {"problem": str, "solution": str, "level": str, "type": str}
            level = int(example.get("level", "3").replace("Level ", ""))

            if difficulty_filter and difficulty_map.get(level) != difficulty_filter:
                continue

            problem = Problem(
                problem_id=f"math_{split}_{idx:05d}",
                domain="math",
                subdomain=example.get("type", "algebra").lower(),
                source="math_dataset",
                difficulty=difficulty_map.get(level, "medium"),
                statement=example["problem"],
                answer_gold=example.get("solution", "").split("\\boxed{")[-1].split("}")[0]
                if "\\boxed{" in example.get("solution", "")
                else None,
                meta_course={
                    "dataset": "math",
                    "split": split,
                    "level": level,
                    "type": example.get("type"),
                },
            )
            problems.append(problem)

        logger.info(f"Ingested {len(problems)} problems from MATH ({split})")
        return problems

    @staticmethod
    def ingest_humaneval(
        max_samples: Optional[int] = None,
    ) -> List[Problem]:
        """
        Ingest HumanEval dataset (code problems).

        Args:
            max_samples: Optional limit

        Returns:
            List of Problem objects
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package required")

        logger.info("Loading HumanEval dataset...")
        dataset = load_dataset("openai_humaneval", split="test")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        problems = []
        for idx, example in enumerate(dataset):
            # HumanEval format: {"task_id": str, "prompt": str, "test": str, ...}
            problem = Problem(
                problem_id=f"humaneval_{example['task_id']}",
                domain="code",
                subdomain="python",
                source="humaneval",
                difficulty="medium",
                statement=example["prompt"],
                answer_gold=example.get("canonical_solution"),
                meta_course={
                    "dataset": "humaneval",
                    "tests": example.get("test", ""),
                    "entry_point": example.get("entry_point"),
                },
            )
            problems.append(problem)

        logger.info(f"Ingested {len(problems)} problems from HumanEval")
        return problems


class CustomDatasetIngestion:
    """Import custom datasets from JSON/CSV."""

    @staticmethod
    def ingest_json_problems(
        json_path: Path,
        source_name: str = "custom",
    ) -> List[Problem]:
        """
        Ingest problems from JSON file.

        Expected format:
        [
          {
            "problem_id": "...",
            "domain": "math|code|logic",
            "statement": "...",
            "answer_gold": "...",
            ...
          },
          ...
        ]

        Args:
            json_path: Path to JSON file
            source_name: Source identifier

        Returns:
            List of Problem objects
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        problems = []
        for item in data:
            # Fill defaults
            item.setdefault("source", source_name)
            item.setdefault("subdomain", None)
            item.setdefault("difficulty", "medium")
            item.setdefault("answer_gold", None)
            item.setdefault("meta_course", None)

            problem = Problem(**item)
            problems.append(problem)

        logger.info(f"Ingested {len(problems)} problems from {json_path}")
        return problems

    @staticmethod
    def ingest_json_cots(
        json_path: Path,
        source_name: str = "custom",
    ) -> List[ChainOfThought]:
        """
        Ingest CoTs from JSON file.

        Expected format:
        [
          {
            "cot_id": "...",
            "problem_id": "...",
            "steps": ["Step 1: ...", "Step 2: ..."],
            "final_answer": "...",
            "is_valid": true,
            ...
          },
          ...
        ]

        Args:
            json_path: Path to JSON file
            source_name: Source identifier

        Returns:
            List of ChainOfThought objects
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cots = []
        for item in data:
            # Fill defaults
            item.setdefault("source", source_name)
            item.setdefault("validation_reason", "")
            item.setdefault("teacher_model", "unknown")

            cot = ChainOfThought(**item)
            cots.append(cot)

        logger.info(f"Ingested {len(cots)} CoTs from {json_path}")
        return cots


class UserInteractionIngestion:
    """Import user interaction logs for continuous learning."""

    @staticmethod
    def ingest_interaction_logs(
        logs_dir: Path,
        pattern: str = "*.ndjson",
        thumbs_up_only: bool = True,
    ) -> tuple[List[Problem], List[ChainOfThought]]:
        """
        Ingest user interaction logs.

        Expected log format (NDJSON):
        {
          "session_id": "...",
          "timestamp": "...",
          "prompt": "...",
          "cot_steps": ["Step 1: ...", ...],
          "final_answer": "...",
          "jepa_score": 0.85,
          "feedback_user": "thumbs_up",
        }

        Args:
            logs_dir: Directory containing log files
            pattern: Glob pattern for log files
            thumbs_up_only: Only import interactions with positive feedback

        Returns:
            (problems, cots) tuple
        """
        log_files = sorted(logs_dir.glob(pattern))

        if not log_files:
            logger.warning(f"No interaction logs found in {logs_dir}")
            return [], []

        problems = []
        cots = []

        for log_file in log_files:
            with open(log_file, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue

                    try:
                        log_entry = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in {log_file}:{line_idx}")
                        continue

                    # Filter by feedback
                    if thumbs_up_only and log_entry.get("feedback_user") != "thumbs_up":
                        continue

                    # Create Problem from user prompt
                    problem_id = f"user_{log_entry['session_id']}"
                    problem = Problem(
                        problem_id=problem_id,
                        domain="mixed",  # User questions can be anything
                        subdomain=None,
                        source="user_interaction",
                        difficulty="unknown",
                        statement=log_entry["prompt"],
                        answer_gold=log_entry.get("final_answer"),
                        meta_course={
                            "session_id": log_entry["session_id"],
                            "timestamp": log_entry.get("timestamp"),
                            "jepa_score": log_entry.get("jepa_score"),
                        },
                    )
                    problems.append(problem)

                    # Create CoT if steps present
                    if log_entry.get("cot_steps"):
                        cot_id = f"user_cot_{log_entry['session_id']}"
                        cot = ChainOfThought(
                            cot_id=cot_id,
                            problem_id=problem_id,
                            steps=log_entry["cot_steps"],
                            final_answer=log_entry.get("final_answer", ""),
                            is_valid=log_entry.get("feedback_user") == "thumbs_up",
                            validation_reason=f"User feedback: {log_entry.get('feedback_user')}",
                            teacher_model="student_llm",
                            source="user_interaction",
                        )
                        cots.append(cot)

        logger.info(
            f"Ingested {len(problems)} problems and {len(cots)} CoTs "
            f"from user interaction logs"
        )

        return problems, cots


def save_problems_to_parquet(problems: List[Problem], output_path: Path) -> None:
    """
    Save problems to parquet file.

    Args:
        problems: List of Problem objects
        output_path: Output parquet path
    """
    from rjepa.utils.io import ParquetIO

    records = [asdict(p) for p in problems]
    ParquetIO.write(records, output_path)


def save_cots_to_parquet(cots: List[ChainOfThought], output_path: Path) -> None:
    """
    Save CoTs to parquet file.

    Args:
        cots: List of ChainOfThought objects
        output_path: Output parquet path
    """
    from rjepa.utils.io import ParquetIO

    records = [asdict(c) for c in cots]
    ParquetIO.write(records, output_path)
