"""
Feedback Collection & Validation Pipeline for Continuous Learning.

Philosophy:
- Not all user interactions are valid training data
- Multi-level validation: JEPA score + user feedback + auto-validation
- Quality > quantity: better to reject ambiguous data than pollute training
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import torch
from datetime import datetime, timedelta

from rjepa.data.user_interactions import UserInteraction
from rjepa.data.schemas import Problem, ChainOfThought
from rjepa.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a user interaction."""

    is_valid: bool
    confidence: float  # 0-1
    validation_method: str  # "jepa_score", "math_check", "code_exec", "manual"
    reason: str
    auto_validated: bool


class FeedbackValidator:
    """
    Validates user interactions for inclusion in training data.

    Multi-level validation:
    1. JEPA score threshold (high coherence)
    2. User feedback (thumbs up)
    3. Auto-validation (math/code if applicable)
    4. Ambiguous cases marked for manual review
    """

    def __init__(
        self,
        jepa_score_threshold: float = 0.7,
        require_user_feedback: bool = True,
        enable_auto_validation: bool = True,
    ):
        """
        Initialize feedback validator.

        Args:
            jepa_score_threshold: Min JEPA score for auto-accept (0-1, higher=stricter)
            require_user_feedback: Require thumbs_up for inclusion
            enable_auto_validation: Run math/code validators when applicable
        """
        self.jepa_score_threshold = jepa_score_threshold
        self.require_user_feedback = require_user_feedback
        self.enable_auto_validation = enable_auto_validation

        logger.info(
            f"FeedbackValidator initialized: jepa_threshold={jepa_score_threshold}, "
            f"require_feedback={require_user_feedback}, auto_validation={enable_auto_validation}"
        )

    def validate_interaction(
        self,
        interaction: UserInteraction,
        domain: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate a single user interaction.

        Decision tree:
        1. If thumbs_down -> REJECT
        2. If thumbs_up + high JEPA score -> ACCEPT
        3. If auto-validation passes (math/code) -> ACCEPT
        4. If ambiguous -> MANUAL_REVIEW

        Args:
            interaction: User interaction to validate
            domain: Problem domain (math, code, logic) - inferred if None

        Returns:
            ValidationResult with decision + confidence
        """
        # 1. Check for explicit rejection
        if interaction.feedback_type == "thumbs_down":
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                validation_method="user_feedback",
                reason="User explicitly marked as incorrect (thumbs_down)",
                auto_validated=True,
            )

        # 2. Check JEPA score (if available)
        jepa_valid = False
        jepa_confidence = 0.0

        if interaction.jepa_score is not None:
            # Convert JEPA-loss to score (lower loss = better)
            # Assume jepa_score is stored as exp(-loss) or similar normalization
            jepa_confidence = interaction.jepa_score
            jepa_valid = jepa_confidence >= self.jepa_score_threshold

        # 3. Check user feedback
        has_thumbs_up = interaction.feedback_type == "thumbs_up"

        # 4. Auto-validation (math/code)
        auto_valid = False
        auto_method = None
        auto_reason = None

        if self.enable_auto_validation and domain:
            if domain == "math":
                auto_valid, auto_reason = self._validate_math(interaction)
                auto_method = "math_check"
            elif domain == "code":
                auto_valid, auto_reason = self._validate_code(interaction)
                auto_method = "code_exec"

        # Decision logic
        if has_thumbs_up and jepa_valid:
            # Best case: user approval + high JEPA score
            return ValidationResult(
                is_valid=True,
                confidence=min(1.0, jepa_confidence + 0.2),  # Boost for user approval
                validation_method="jepa_score+user_feedback",
                reason=f"High JEPA score ({jepa_confidence:.2f}) + user approval",
                auto_validated=True,
            )

        elif auto_valid:
            # Auto-validation passed (math/code correct)
            return ValidationResult(
                is_valid=True,
                confidence=0.9,
                validation_method=auto_method,
                reason=auto_reason,
                auto_validated=True,
            )

        elif has_thumbs_up and jepa_confidence >= 0.5:
            # User approval but medium JEPA score -> accept with lower confidence
            return ValidationResult(
                is_valid=True,
                confidence=0.7,
                validation_method="user_feedback",
                reason=f"User approval (JEPA score={jepa_confidence:.2f}, medium)",
                auto_validated=True,
            )

        elif jepa_valid and not self.require_user_feedback:
            # High JEPA score alone (if user feedback not required)
            return ValidationResult(
                is_valid=True,
                confidence=jepa_confidence,
                validation_method="jepa_score",
                reason=f"High JEPA score ({jepa_confidence:.2f}), no user feedback required",
                auto_validated=True,
            )

        else:
            # Ambiguous case -> reject or manual review
            return ValidationResult(
                is_valid=False,
                confidence=0.3,
                validation_method="ambiguous",
                reason=f"Insufficient evidence (JEPA={jepa_confidence:.2f}, feedback={interaction.feedback_type})",
                auto_validated=False,
            )

    def _validate_math(self, interaction: UserInteraction) -> Tuple[bool, str]:
        """
        Validate math problem (symbolic/numeric check).

        Returns:
            (is_valid, reason)
        """
        try:
            # Extract final answer from response
            # Simple heuristic: look for "= X" or "answer is X"
            import re

            response = interaction.response.lower()

            # Pattern: "= number" or "answer is number"
            pattern = r"(?:=\s*|answer\s+is\s+)(-?\d+(?:\.\d+)?)"
            match = re.search(pattern, response)

            if match:
                computed_answer = float(match.group(1))

                # TODO: Compare with ground truth if available
                # For now, just check if numeric answer was computed
                return True, f"Numeric answer computed: {computed_answer}"

            return False, "No numeric answer found in response"

        except Exception as e:
            logger.warning(f"Math validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    def _validate_code(self, interaction: UserInteraction) -> Tuple[bool, str]:
        """
        Validate code problem (execution test).

        Returns:
            (is_valid, reason)
        """
        try:
            # Extract code from response (look for ```python blocks)
            import re

            pattern = r"```python\n(.*?)```"
            match = re.search(pattern, interaction.response, re.DOTALL)

            if match:
                code = match.group(1)

                # TODO: Execute in sandbox with timeout
                # For now, just check if syntactically valid
                compile(code, "<string>", "exec")

                return True, "Code is syntactically valid"

            return False, "No Python code block found"

        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            logger.warning(f"Code validation failed: {e}")
            return False, f"Validation error: {str(e)}"


class FeedbackPipeline:
    """
    End-to-end pipeline: load interactions -> validate -> convert to training data.

    Workflow:
    1. Load interactions from logs (JSONL)
    2. Validate each interaction (multi-level)
    3. Convert validated interactions to (Problem, CoT) pairs
    4. Save to data/datasets/ (versioned)
    5. Track statistics (acceptance rate, etc.)
    """

    def __init__(
        self,
        log_dir: Path,
        output_dir: Path,
        validator: FeedbackValidator,
    ):
        """
        Initialize feedback pipeline.

        Args:
            log_dir: Directory with interaction logs (JSONL files)
            output_dir: Output directory for validated datasets
            validator: FeedbackValidator instance
        """
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.validator = validator

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"FeedbackPipeline initialized: log_dir={log_dir}, output_dir={output_dir}")

    def load_interactions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        opted_in_only: bool = True,
    ) -> List[UserInteraction]:
        """
        Load interactions from JSONL logs.

        Args:
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date (YYYY-MM-DD)
            opted_in_only: Only load opted-in interactions

        Returns:
            List of UserInteraction objects
        """
        import json

        # Find all log files in date range
        log_files = sorted(self.log_dir.glob("interactions_*.jsonl"))

        if start_date or end_date:
            # TODO: Filter by date range
            pass

        # Load all JSONL files
        interactions = []
        for log_file in log_files:
            logger.info(f"Loading {log_file}...")

            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    interaction = UserInteraction(**data)

                    # Filter by opt-in
                    if opted_in_only and not interaction.opted_in:
                        continue

                    interactions.append(interaction)

        logger.info(f"Loaded {len(interactions)} interactions from {len(log_files)} files")

        return interactions

    def validate_batch(
        self,
        interactions: List[UserInteraction],
    ) -> Tuple[List[UserInteraction], List[ValidationResult]]:
        """
        Validate a batch of interactions.

        Returns:
            (valid_interactions, validation_results)
        """
        valid = []
        results = []

        for interaction in interactions:
            # Infer domain from interaction metadata
            domain = interaction.domain if hasattr(interaction, "domain") else None

            result = self.validator.validate_interaction(interaction, domain)
            results.append(result)

            if result.is_valid:
                valid.append(interaction)

        logger.info(
            f"Validation complete: {len(valid)}/{len(interactions)} accepted "
            f"({100*len(valid)/len(interactions):.1f}%)"
        )

        return valid, results

    def convert_to_training_data(
        self,
        interactions: List[UserInteraction],
        llm: Optional[LLMAdapter] = None,
    ) -> Tuple[List[Problem], List[ChainOfThought]]:
        """
        Convert validated interactions to (Problem, CoT) pairs.

        Args:
            interactions: Validated interactions
            llm: LLMAdapter for re-extracting latents (optional)

        Returns:
            (problems, cots)
        """
        problems = []
        cots = []

        for idx, interaction in enumerate(interactions):
            # Create Problem from user prompt
            problem = Problem(
                problem_id=f"user_{interaction.session_id}_{idx}",
                domain=interaction.domain if hasattr(interaction, "domain") else "general",
                subdomain="user_interaction",
                source="user_feedback",
                difficulty="unknown",
                statement=interaction.prompt,
                answer_gold=None,  # Not available for user interactions
                meta_course=None,
            )

            # Create CoT from response steps
            cot = ChainOfThought(
                cot_id=interaction.interaction_id,
                problem_id=problem.problem_id,
                steps=interaction.cot_steps,
                final_answer=interaction.response.split("\n")[-1],  # Last line as answer
                is_valid=interaction.is_validated,
                validation_reason=interaction.validation_method or "user_feedback",
                teacher_model=f"user+rjepa_{interaction.jepa_mode}",
                source="user_feedback",  # Required field
                meta={
                    "jepa_score": interaction.jepa_score,
                    "feedback_type": interaction.feedback_type,
                    "timestamp": interaction.timestamp,
                },
            )

            problems.append(problem)
            cots.append(cot)

        logger.info(f"Converted {len(interactions)} interactions to training data")

        return problems, cots

    def save_dataset(
        self,
        problems: List[Problem],
        cots: List[ChainOfThought],
        version: str,
    ):
        """
        Save validated dataset to parquet (versioned).

        Args:
            problems: List of Problem objects
            cots: List of CoT objects
            version: Dataset version (e.g., "v1.3.0-user-feedback")
        """
        # Create version directory
        version_dir = self.output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save problems (use model_dump() for Pydantic models)
        problems_df = pd.DataFrame([p.model_dump() for p in problems])
        problems_path = version_dir / "problems.parquet"
        problems_df.to_parquet(problems_path, index=False)

        # Save CoTs (use model_dump() for Pydantic models)
        cots_df = pd.DataFrame([c.model_dump() for c in cots])
        cots_path = version_dir / "cots.parquet"
        cots_df.to_parquet(cots_path, index=False)

        # Save metadata
        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "num_problems": len(problems),
            "num_cots": len(cots),
            "source": "user_feedback",
        }

        import json
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Saved dataset {version}: {len(problems)} problems, {len(cots)} CoTs "
            f"to {version_dir}"
        )

    def run_pipeline(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Run full pipeline: load -> validate -> convert -> save.

        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            version: Dataset version (auto-generated if None)

        Returns:
            Statistics dict
        """
        # Load interactions
        interactions = self.load_interactions(start_date, end_date, opted_in_only=True)

        if len(interactions) == 0:
            logger.warning("No interactions found, skipping pipeline")
            return {"num_loaded": 0}

        # Validate
        valid_interactions, validation_results = self.validate_batch(interactions)

        # Convert to training data
        problems, cots = self.convert_to_training_data(valid_interactions)

        # Generate version if not provided
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            version = f"user-feedback-{timestamp}"

        # Save
        self.save_dataset(problems, cots, version)

        # Statistics
        stats = {
            "num_loaded": len(interactions),
            "num_valid": len(valid_interactions),
            "num_rejected": len(interactions) - len(valid_interactions),
            "acceptance_rate": len(valid_interactions) / len(interactions),
            "version": version,
        }

        logger.info(f"Pipeline complete: {stats}")

        return stats


def create_feedback_pipeline(
    log_dir: str = "logs/interactions",
    output_dir: str = "data/datasets/user-feedback",
    jepa_score_threshold: float = 0.7,
) -> FeedbackPipeline:
    """
    Factory function to create FeedbackPipeline.

    Args:
        log_dir: Directory with interaction logs
        output_dir: Output directory for validated datasets
        jepa_score_threshold: Min JEPA score for auto-accept

    Returns:
        FeedbackPipeline instance
    """
    validator = FeedbackValidator(
        jepa_score_threshold=jepa_score_threshold,
        require_user_feedback=True,
        enable_auto_validation=True,
    )

    return FeedbackPipeline(
        log_dir=Path(log_dir),
        output_dir=Path(output_dir),
        validator=validator,
    )
