"""
User Interactions Logging for Continuous Learning.

Philosophy:
- Every user interaction = potential learning opportunity
- Log: prompt, response, CoT steps, JEPA score, feedback
- Privacy-first: anonymization, opt-in, PII filtering
- Enable continuous improvement of R-JEPA world model
"""
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import json
import hashlib
import pandas as pd
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class UserInteraction:
    """Single user interaction record."""

    # Required fields (no defaults)
    session_id: str
    interaction_id: str
    prompt: str
    response: str
    cot_steps: List[str]  # Structured reasoning steps
    jepa_mode: str  # "off", "rerank", "nudge", "plan", "guidance"
    timestamp: str

    # Optional fields (with defaults)
    user_id_hash: Optional[str] = None  # SHA256 hash, not raw user_id
    jepa_score: Optional[float] = None  # JEPA-loss or confidence
    jepa_candidates: Optional[int] = None  # Number of candidates (rerank mode)
    feedback_type: Optional[str] = None  # "thumbs_up", "thumbs_down", "comment"
    feedback_comment: Optional[str] = None
    is_validated: bool = False  # Auto-validation passed (math/code)
    validation_method: Optional[str] = None  # "math", "code", "manual", None
    domain: Optional[str] = None  # "math", "code", "logic", etc.
    model_version: str = "rjepa-v1.0.0"  # R-JEPA checkpoint version
    pii_filtered: bool = False  # PII filtering applied
    opted_in: bool = False  # User consented to use for training


class InteractionLogger:
    """
    Logs user interactions to disk for continuous learning.

    Philosophy: Privacy-first, opt-in, anonymized.
    """

    def __init__(
        self,
        log_dir: Path,
        enable_pii_filter: bool = True,
        auto_flush_interval: int = 100,
    ):
        """
        Initialize interaction logger.

        Args:
            log_dir: Directory to store interaction logs
            enable_pii_filter: Enable PII filtering (recommended)
            auto_flush_interval: Flush to disk every N interactions
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.enable_pii_filter = enable_pii_filter
        self.auto_flush_interval = auto_flush_interval

        # In-memory buffer
        self.buffer: List[UserInteraction] = []

        # Current log file (daily rotation)
        self.current_log_file = self._get_daily_log_file()

        logger.info(
            f"InteractionLogger initialized: log_dir={log_dir}, "
            f"pii_filter={enable_pii_filter}"
        )

    def _get_daily_log_file(self) -> Path:
        """Get log file path for today (daily rotation)."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"interactions_{today}.jsonl"

    def _anonymize_user_id(self, user_id: Optional[str]) -> Optional[str]:
        """Hash user_id for anonymization."""
        if user_id is None:
            return None
        return hashlib.sha256(user_id.encode()).hexdigest()

    def _filter_pii(self, text: str) -> str:
        """
        Filter PII from text (basic implementation).

        TODO: Use proper NER model for production.
        """
        if not self.enable_pii_filter:
            return text

        # Basic patterns (extend with NER model in production)
        import re

        # Email
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)

        # Phone (US format)
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)

        # SSN
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)

        # Credit card (simple)
        text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD]", text)

        return text

    def log_interaction(
        self,
        session_id: str,
        prompt: str,
        response: str,
        cot_steps: List[str],
        jepa_mode: str,
        user_id: Optional[str] = None,
        jepa_score: Optional[float] = None,
        jepa_candidates: Optional[int] = None,
        feedback_type: Optional[str] = None,
        feedback_comment: Optional[str] = None,
        domain: Optional[str] = None,
        opted_in: bool = False,
    ) -> str:
        """
        Log a single user interaction.

        Args:
            session_id: Session ID (unique per conversation)
            prompt: User's input prompt
            response: System's response
            cot_steps: Reasoning steps (structured)
            jepa_mode: JEPA mode used ("off", "rerank", "nudge", etc.)
            user_id: User ID (will be hashed for anonymization)
            jepa_score: JEPA score/loss
            jepa_candidates: Number of candidates (rerank mode)
            feedback_type: User feedback type
            feedback_comment: User feedback comment
            domain: Problem domain
            opted_in: User consented to use for training

        Returns:
            interaction_id: Unique interaction ID
        """
        # Generate interaction ID
        interaction_id = hashlib.sha256(
            f"{session_id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # Anonymize user ID
        user_id_hash = self._anonymize_user_id(user_id)

        # Filter PII
        prompt_filtered = self._filter_pii(prompt)
        response_filtered = self._filter_pii(response)
        cot_steps_filtered = [self._filter_pii(step) for step in cot_steps]
        feedback_comment_filtered = (
            self._filter_pii(feedback_comment) if feedback_comment else None
        )

        # Create interaction record
        interaction = UserInteraction(
            session_id=session_id,
            interaction_id=interaction_id,
            user_id_hash=user_id_hash,
            prompt=prompt_filtered,
            response=response_filtered,
            cot_steps=cot_steps_filtered,
            jepa_mode=jepa_mode,
            jepa_score=jepa_score,
            jepa_candidates=jepa_candidates,
            feedback_type=feedback_type,
            feedback_comment=feedback_comment_filtered,
            timestamp=datetime.now().isoformat(),
            domain=domain,
            pii_filtered=self.enable_pii_filter,
            opted_in=opted_in,
        )

        # Add to buffer
        self.buffer.append(interaction)

        # Auto-flush if buffer is full
        if len(self.buffer) >= self.auto_flush_interval:
            self.flush()

        logger.debug(f"Logged interaction: {interaction_id}")

        return interaction_id

    def update_feedback(
        self,
        interaction_id: str,
        feedback_type: str,
        feedback_comment: Optional[str] = None,
    ):
        """
        Update feedback for an existing interaction.

        Args:
            interaction_id: Interaction ID to update
            feedback_type: Feedback type ("thumbs_up", "thumbs_down", "comment")
            feedback_comment: Optional comment
        """
        # Find in buffer
        for interaction in self.buffer:
            if interaction.interaction_id == interaction_id:
                interaction.feedback_type = feedback_type
                interaction.feedback_comment = (
                    self._filter_pii(feedback_comment) if feedback_comment else None
                )
                logger.debug(f"Updated feedback for {interaction_id}: {feedback_type}")
                return

        # If not in buffer, it's already flushed
        # Would need to reload from disk to update (complex, defer to batch processing)
        logger.warning(
            f"Interaction {interaction_id} not in buffer, "
            f"feedback update deferred to batch processing"
        )

    def mark_validated(
        self,
        interaction_id: str,
        validation_method: str,
        is_validated: bool,
    ):
        """
        Mark interaction as validated (auto or manual).

        Args:
            interaction_id: Interaction ID
            validation_method: "math", "code", "manual"
            is_validated: Validation passed
        """
        for interaction in self.buffer:
            if interaction.interaction_id == interaction_id:
                interaction.is_validated = is_validated
                interaction.validation_method = validation_method
                return

    def flush(self):
        """Flush buffer to disk (JSONL format)."""
        if not self.buffer:
            return

        # Check if we need to rotate to a new file (daily)
        log_file = self._get_daily_log_file()

        # Append to JSONL file
        with open(log_file, "a", encoding="utf-8") as f:
            for interaction in self.buffer:
                f.write(json.dumps(asdict(interaction)) + "\n")

        logger.info(f"Flushed {len(self.buffer)} interactions to {log_file}")

        # Clear buffer
        self.buffer.clear()

    def load_interactions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        opted_in_only: bool = True,
    ) -> pd.DataFrame:
        """
        Load interactions from disk.

        Args:
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date (YYYY-MM-DD)
            opted_in_only: Only load interactions with opted_in=True

        Returns:
            DataFrame of interactions
        """
        # Find all log files in date range
        log_files = []

        if start_date and end_date:
            # TODO: Implement date range filtering
            log_files = list(self.log_dir.glob("interactions_*.jsonl"))
        else:
            log_files = list(self.log_dir.glob("interactions_*.jsonl"))

        # Load all JSONL files
        interactions = []
        for log_file in log_files:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    interaction = json.loads(line)
                    interactions.append(interaction)

        # Convert to DataFrame
        df = pd.DataFrame(interactions)

        # Filter by opted_in
        if opted_in_only and len(df) > 0:
            df = df[df["opted_in"] == True]

        logger.info(f"Loaded {len(df)} interactions from {len(log_files)} files")

        return df


def create_interaction_logger(
    log_dir: str = "logs/interactions",
    enable_pii_filter: bool = True,
) -> InteractionLogger:
    """
    Factory function to create InteractionLogger.

    Args:
        log_dir: Directory for interaction logs
        enable_pii_filter: Enable PII filtering

    Returns:
        InteractionLogger instance
    """
    return InteractionLogger(
        log_dir=Path(log_dir),
        enable_pii_filter=enable_pii_filter,
    )
