"""
Teacher Problem & CoT Generator.

Generates structured problems and Chain-of-Thought solutions.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import yaml
from pathlib import Path

from rjepa.teacher.client import TeacherClient
from rjepa.data.schemas import Problem, ChainOfThought

logger = logging.getLogger(__name__)


class ProblemGenerator:
    """
    Generate academic problems via teacher LLM.

    Supports multiple domains: math, code, logic.
    """

    def __init__(
        self,
        client: TeacherClient,
        prompts_config_path: str = "configs/teacher/prompts.yaml"
    ):
        """
        Initialize Problem Generator.

        Args:
            client: Teacher LLM client
            prompts_config_path: Path to prompts config YAML
        """
        self.client = client
        self.prompts_config_path = prompts_config_path

        # Load prompts
        with open(prompts_config_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

        logger.info(f"Problem generator initialized with {client.model}")

    def generate_problems(
        self,
        domain: str,
        subdomain: Optional[str] = None,
        difficulty: str = "medium",
        num_problems: int = 10,
    ) -> List[Problem]:
        """
        Generate a batch of problems.

        Args:
            domain: Problem domain ("math", "code", "logic")
            subdomain: Optional subdomain (e.g., "algebra", "sorting")
            difficulty: Difficulty level ("easy", "medium", "hard")
            num_problems: Number of problems to generate

        Returns:
            List of Problem objects
        """
        # Get prompt template
        template = self.prompts["problem_generation"].get(domain)
        if not template:
            raise ValueError(f"No prompt template for domain: {domain}")

        # Format prompt
        prompt = template.format(
            domain=domain,
            subdomain=subdomain or "general",
            difficulty=difficulty,
            num=num_problems,
        )

        # Generate
        try:
            response = self.client.generate_json(prompt)

            # Parse problems
            problems = []
            for i, problem_dict in enumerate(response.get("problems", [])):
                problem = Problem(
                    problem_id=f"{domain}_{subdomain or 'gen'}_{i}_{datetime.now().timestamp()}",
                    domain=domain,
                    subdomain=subdomain or problem_dict.get("subdomain", "general"),
                    source=f"teacher_{self.client.model}",
                    difficulty=difficulty,
                    statement=problem_dict["statement"],
                    answer_gold=problem_dict.get("answer"),
                    meta_course=problem_dict.get("notions"),
                )
                problems.append(problem)

            logger.info(f"Generated {len(problems)} problems for {domain}/{subdomain}")
            return problems

        except Exception as e:
            logger.error(f"Problem generation failed: {e}", exc_info=True)
            return []


class CoTGenerator:
    """
    Generate Chain-of-Thought solutions for problems.
    """

    def __init__(
        self,
        client: TeacherClient,
        prompts_config_path: str = "configs/teacher/prompts.yaml"
    ):
        """
        Initialize CoT Generator.

        Args:
            client: Teacher LLM client
            prompts_config_path: Path to prompts config YAML
        """
        self.client = client
        self.prompts_config_path = prompts_config_path

        # Load prompts
        with open(prompts_config_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

        logger.info(f"CoT generator initialized with {client.model}")

    def generate_cot(
        self,
        problem: Problem,
        num_samples: int = 3,
        temperature: float = 0.8,
    ) -> List[ChainOfThought]:
        """
        Generate Chain-of-Thought solutions for a problem.

        Args:
            problem: Problem to solve
            num_samples: Number of CoT samples to generate
            temperature: Sampling temperature (higher = more diversity)

        Returns:
            List of ChainOfThought objects
        """
        # Get prompt template
        template = self.prompts["cot_generation"].get(problem.domain)
        if not template:
            template = self.prompts["cot_generation"]["default"]

        # Format prompt
        prompt = template.format(
            statement=problem.statement,
            num_samples=num_samples,
        )

        # Generate multiple samples
        cots = []
        for i in range(num_samples):
            try:
                response = self.client.generate_json(
                    prompt=prompt,
                    temperature=temperature,
                )

                # Parse steps
                steps = response.get("steps", [])
                final_answer = response.get("final_answer", "")

                # Create CoT object
                cot = ChainOfThought(
                    cot_id=f"{problem.problem_id}_cot_{i}",
                    problem_id=problem.problem_id,
                    steps=steps,
                    final_answer=final_answer,
                    is_valid=False,  # Will be validated separately
                    validation_reason="",
                    teacher_model=self.client.model,
                    source=f"teacher_{self.client.model}",
                )
                cots.append(cot)

            except Exception as e:
                logger.error(f"CoT generation failed (sample {i}): {e}")

        logger.info(f"Generated {len(cots)} CoT samples for problem {problem.problem_id}")
        return cots

    def generate_cot_diverse(
        self,
        problem: Problem,
        clients: List[TeacherClient],
        num_per_client: int = 2,
    ) -> List[ChainOfThought]:
        """
        Generate diverse CoT from multiple teacher clients.

        Args:
            problem: Problem to solve
            clients: List of teacher clients
            num_per_client: Number of CoT per client

        Returns:
            List of ChainOfThought objects from all clients
        """
        all_cots = []

        for client in clients:
            # Temporarily use this client
            original_client = self.client
            self.client = client

            cots = self.generate_cot(problem, num_samples=num_per_client)
            all_cots.extend(cots)

            # Restore original client
            self.client = original_client

        logger.info(f"Generated {len(all_cots)} diverse CoT samples from {len(clients)} clients")
        return all_cots


class DatasetGenerator:
    """
    High-level dataset generator (problems + CoT + validation).
    """

    def __init__(
        self,
        problem_generator: ProblemGenerator,
        cot_generator: CoTGenerator,
    ):
        """
        Initialize Dataset Generator.

        Args:
            problem_generator: Problem generator instance
            cot_generator: CoT generator instance
        """
        self.problem_generator = problem_generator
        self.cot_generator = cot_generator

        logger.info("Dataset generator initialized")

    def generate_dataset(
        self,
        domain: str,
        num_problems: int = 100,
        cots_per_problem: int = 3,
        difficulty: str = "medium",
        subdomain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete dataset (problems + CoT).

        Args:
            domain: Problem domain
            num_problems: Number of problems to generate
            cots_per_problem: CoT samples per problem
            difficulty: Difficulty level
            subdomain: Optional subdomain

        Returns:
            Dict with "problems" and "cots" lists
        """
        logger.info(f"Generating dataset: {num_problems} problems x {cots_per_problem} CoT")

        # Generate problems
        problems = self.problem_generator.generate_problems(
            domain=domain,
            subdomain=subdomain,
            difficulty=difficulty,
            num_problems=num_problems,
        )

        # Generate CoT for each problem
        all_cots = []
        for i, problem in enumerate(problems):
            logger.info(f"Generating CoT for problem {i+1}/{len(problems)}")

            cots = self.cot_generator.generate_cot(
                problem=problem,
                num_samples=cots_per_problem,
            )
            all_cots.extend(cots)

        logger.info(f"Dataset generated: {len(problems)} problems, {len(all_cots)} CoT")

        return {
            "problems": problems,
            "cots": all_cots,
            "metadata": {
                "domain": domain,
                "subdomain": subdomain,
                "difficulty": difficulty,
                "num_problems": len(problems),
                "num_cots": len(all_cots),
                "generated_at": datetime.now().isoformat(),
            }
        }
