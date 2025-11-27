"""
R-JEPA Data Package.
"""
from .schemas import (
    Problem,
    ChainOfThought,
    LatentSequence,
    DatasetVersion,
    RJEPACheckpoint,
)
from .user_interactions import (
    UserInteraction,
    InteractionLogger,
    create_interaction_logger,
)
from .feedback_pipeline import (
    ValidationResult,
    FeedbackValidator,
    FeedbackPipeline,
    create_feedback_pipeline,
)

__all__ = [
    # Schemas
    "Problem",
    "ChainOfThought",
    "LatentSequence",
    "DatasetVersion",
    "RJEPACheckpoint",
    # User interactions (Phase 15)
    "UserInteraction",
    "InteractionLogger",
    "create_interaction_logger",
    # Feedback pipeline (Phase 15)
    "ValidationResult",
    "FeedbackValidator",
    "FeedbackPipeline",
    "create_feedback_pipeline",
]
