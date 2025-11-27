"""
Import Academic Datasets (Free, Open-Source)

Datasets supportés:
- GSM8K: Grade School Math (MIT License)
- MATH: Competition Math (MIT License)
- HumanEval: Code Generation (MIT License)
- MBPP: Python Programming (Apache 2.0)
- ARC: AI2 Reasoning Challenge (Apache 2.0)
- MMLU: Massive Multitask Language Understanding (MIT License)

Usage:
    python -m rjepa.data.import_academic --dataset gsm8k --output data/datasets/academic/
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class Problem(BaseModel):
    """Schema for a problem"""
    problem_id: str
    domain: str  # "math", "code", "logic"
    subdomain: str
    source: str
    difficulty: str  # "easy", "medium", "hard"
    statement: str
    answer_gold: Optional[str] = None
    meta: Optional[Dict] = None


class ChainOfThought(BaseModel):
    """Schema for a chain of thought"""
    cot_id: str
    problem_id: str
    steps: List[str]
    final_answer: str
    is_valid: bool
    validation_reason: str
    teacher_model: str = "dataset_original"
    meta: Optional[Dict] = None


# ═══════════════════════════════════════════════════════════════════════════
# GSM8K (Grade School Math - 8,500 problems)
# ═══════════════════════════════════════════════════════════════════════════

class GSM8KImporter:
    """
    Import GSM8K dataset

    Source: https://github.com/openai/grade-school-math
    License: MIT
    Size: 8,500 problems (7,500 train / 1,000 test)
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "math" / "gsm8k"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def import_dataset(self) -> Dict[str, int]:
        """Import GSM8K from HuggingFace"""
        logger.info("Downloading GSM8K dataset...")

        # Load from HuggingFace
        dataset = load_dataset("openai/gsm8k", "main")

        stats = {}

        for split in ["train", "test"]:
            logger.info(f"Processing GSM8K {split}...")

            problems = []
            cots = []

            for idx, example in enumerate(dataset[split]):
                # Extract problem
                problem_id = f"gsm8k_{split}_{idx}"

                problem = Problem(
                    problem_id=problem_id,
                    domain="math",
                    subdomain="arithmetic_word_problems",
                    source="gsm8k",
                    difficulty="easy",  # Grade school level
                    statement=example["question"],
                    answer_gold=self._extract_answer(example["answer"]),
                    meta={"original_answer_text": example["answer"]}
                )
                problems.append(problem)

                # Extract CoT
                steps = self._parse_solution_steps(example["answer"])

                cot = ChainOfThought(
                    cot_id=f"{problem_id}_cot_0",
                    problem_id=problem_id,
                    steps=steps,
                    final_answer=problem.answer_gold,
                    is_valid=True,  # GSM8K solutions are human-verified
                    validation_reason="dataset_verified",
                    teacher_model="gsm8k_human",
                    meta={"source": "gsm8k"}
                )
                cots.append(cot)

            # Save to parquet
            self._save_split(problems, cots, split)
            stats[split] = len(problems)

            logger.info(f"[OK] GSM8K {split}: {len(problems)} problems")

        return stats

    def _extract_answer(self, answer_text: str) -> str:
        """Extract final numerical answer from GSM8K answer"""
        # GSM8K format: "...#### 42"
        if "####" in answer_text:
            return answer_text.split("####")[-1].strip()
        return answer_text.strip()

    def _parse_solution_steps(self, answer_text: str) -> List[str]:
        """Parse GSM8K solution into steps"""
        # Split by sentence/line
        lines = [line.strip() for line in answer_text.split("\n") if line.strip()]

        # Remove final answer line (####)
        steps = [line for line in lines if not line.startswith("####")]

        # Add step numbering if not present
        if steps and not steps[0].lower().startswith("step"):
            steps = [f"Step {i+1}: {step}" for i, step in enumerate(steps)]

        return steps

    def _save_split(self, problems: List[Problem], cots: List[ChainOfThought], split: str):
        """Save problems and CoTs to JSON"""
        # Save problems
        problems_file = self.output_dir / f"{split}_problems.json"
        with open(problems_file, "w") as f:
            json.dump([p.model_dump() for p in problems], f, indent=2)

        # Save CoTs
        cots_file = self.output_dir / f"{split}_cots.json"
        with open(cots_file, "w") as f:
            json.dump([c.model_dump() for c in cots], f, indent=2)

        logger.info(f"Saved to {self.output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# MATH (Competition Math - 12,500 problems)
# ═══════════════════════════════════════════════════════════════════════════

class MATHImporter:
    """
    Import MATH dataset

    Source: https://github.com/hendrycks/math
    License: MIT
    Size: 12,500 problems
    Domains: Algebra, Geometry, Number Theory, Counting, Probability, etc.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "math" / "competition_math"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def import_dataset(self) -> Dict[str, int]:
        """Import MATH from HuggingFace"""
        logger.info("Downloading MATH dataset...")

        # Load from HuggingFace (using EleutherAI mirror with all subdomains)
        # Available configs: algebra, counting_and_probability, geometry,
        # intermediate_algebra, number_theory, prealgebra, precalculus
        configs = [
            'algebra', 'counting_and_probability', 'geometry',
            'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
        ]

        stats = {}

        for split in ["train", "test"]:
            logger.info(f"Processing MATH {split}...")

            problems = []
            cots = []
            idx = 0

            # Load each subdomain
            for config in configs:
                logger.info(f"  Loading {config}...")
                dataset = load_dataset("EleutherAI/hendrycks_math", config)

                if split not in dataset:
                    continue

                for example in dataset[split]:
                    problem_id = f"math_{split}_{idx}"
                    idx += 1

                    # Map difficulty level
                    difficulty_map = {"Level 1": "easy", "Level 2": "easy",
                                      "Level 3": "medium", "Level 4": "medium",
                                      "Level 5": "hard"}
                    difficulty = difficulty_map.get(example.get("level", ""), "medium")

                    problem = Problem(
                        problem_id=problem_id,
                        domain="math",
                        subdomain=config,
                        source="competition_math",
                        difficulty=difficulty,
                        statement=example["problem"],
                        answer_gold=example["solution"],
                        meta={
                            "level": example.get("level", ""),
                            "type": config
                        }
                    )
                    problems.append(problem)

                    # Extract CoT from solution
                    steps = self._parse_latex_solution(example["solution"])

                    cot = ChainOfThought(
                        cot_id=f"{problem_id}_cot_0",
                        problem_id=problem_id,
                        steps=steps,
                        final_answer=example["solution"].split("boxed{")[-1].split("}")[0] if "boxed{" in example["solution"] else "",
                        is_valid=True,
                        validation_reason="dataset_verified",
                        teacher_model="math_human",
                        meta={"latex_solution": example["solution"]}
                    )
                    cots.append(cot)

            self._save_split(problems, cots, split)
            stats[split] = len(problems)

            logger.info(f"[OK] MATH {split}: {len(problems)} problems")

        return stats

    def _parse_latex_solution(self, solution: str) -> List[str]:
        """Parse MATH solution (LaTeX) into steps"""
        # Simple heuristic: split by double newline or major transitions
        import re

        # Remove excessive whitespace
        solution = re.sub(r'\n\s*\n', '\n\n', solution)

        # Split by paragraph or sentence
        paragraphs = [p.strip() for p in solution.split('\n\n') if p.strip()]

        # Add step numbering
        steps = [f"Step {i+1}: {para}" for i, para in enumerate(paragraphs)]

        return steps

    def _save_split(self, problems: List[Problem], cots: List[ChainOfThought], split: str):
        """Save to JSON"""
        problems_file = self.output_dir / f"{split}_problems.json"
        with open(problems_file, "w", encoding="utf-8") as f:
            json.dump([p.model_dump() for p in problems], f, indent=2, ensure_ascii=False)

        cots_file = self.output_dir / f"{split}_cots.json"
        with open(cots_file, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in cots], f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# HumanEval (Code Generation - 164 problems)
# ═══════════════════════════════════════════════════════════════════════════

class HumanEvalImporter:
    """
    Import HumanEval dataset

    Source: https://github.com/openai/human-eval
    License: MIT
    Size: 164 Python programming problems
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "code" / "humaneval"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def import_dataset(self) -> Dict[str, int]:
        """Import HumanEval from HuggingFace"""
        logger.info("Downloading HumanEval dataset...")

        # Load from HuggingFace
        dataset = load_dataset("openai_humaneval")

        problems = []
        cots = []

        for idx, example in enumerate(dataset["test"]):
            problem_id = f"humaneval_{idx}"

            problem = Problem(
                problem_id=problem_id,
                domain="code",
                subdomain="python_programming",
                source="humaneval",
                difficulty="medium",  # Competition level
                statement=example["prompt"],
                answer_gold=example["canonical_solution"],
                meta={
                    "task_id": example["task_id"],
                    "entry_point": example["entry_point"],
                    "test": example["test"]
                }
            )
            problems.append(problem)

            # Extract CoT (convert code to explained steps)
            steps = self._code_to_steps(example["canonical_solution"])

            cot = ChainOfThought(
                cot_id=f"{problem_id}_cot_0",
                problem_id=problem_id,
                steps=steps,
                final_answer=example["canonical_solution"],
                is_valid=True,
                validation_reason="passes_unit_tests",
                teacher_model="humaneval_canonical",
                meta={"entry_point": example["entry_point"]}
            )
            cots.append(cot)

        self._save_split(problems, cots, "test")

        logger.info(f"[OK] HumanEval: {len(problems)} problems")

        return {"test": len(problems)}

    def _code_to_steps(self, code: str) -> List[str]:
        """Convert code to reasoning steps"""
        # Simple heuristic: one step per significant line
        lines = [line.strip() for line in code.split("\n") if line.strip() and not line.strip().startswith("#")]

        steps = [f"Step {i+1}: {line}" for i, line in enumerate(lines)]

        return steps

    def _save_split(self, problems: List[Problem], cots: List[ChainOfThought], split: str):
        """Save to JSON"""
        problems_file = self.output_dir / f"{split}_problems.json"
        with open(problems_file, "w") as f:
            json.dump([p.model_dump() for p in problems], f, indent=2)

        cots_file = self.output_dir / f"{split}_cots.json"
        with open(cots_file, "w") as f:
            json.dump([c.model_dump() for c in cots], f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def import_all_datasets(output_dir: str = "data/datasets/academic"):
    """Import all academic datasets"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("IMPORTING ACADEMIC DATASETS")
    logger.info("=" * 80)

    total_stats = {}

    # GSM8K
    logger.info("\n[1/3] GSM8K (Grade School Math)")
    gsm8k = GSM8KImporter(output_path)
    total_stats["gsm8k"] = gsm8k.import_dataset()

    # MATH
    logger.info("\n[2/3] MATH (Competition Math)")
    math = MATHImporter(output_path)
    total_stats["math"] = math.import_dataset()

    # HumanEval
    logger.info("\n[3/3] HumanEval (Code Generation)")
    humaneval = HumanEvalImporter(output_path)
    total_stats["humaneval"] = humaneval.import_dataset()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("IMPORT COMPLETE!")
    logger.info("=" * 80)

    total_problems = sum(sum(splits.values()) for splits in total_stats.values())
    logger.info(f"\nTotal problems imported: {total_problems}")

    for dataset, splits in total_stats.items():
        logger.info(f"  {dataset}: {sum(splits.values())} problems")
        for split, count in splits.items():
            logger.info(f"    - {split}: {count}")

    logger.info(f"\nDatasets saved to: {output_path}")

    return total_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import academic datasets")
    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/academic",
        help="Output directory"
    )

    args = parser.parse_args()

    import_all_datasets(args.output)
