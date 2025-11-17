"""
R-JEPA Inference Modes.

Three exploitation modes:
- Re-ranking: Choose best CoT among N candidates
- Nudge: Correct latents in real-time
- Plan: Complete missing steps
"""
from .rerank import (
    rerank_cots_with_jepa,
    rerank_existing_cots,
    rerank_with_ensembling,
)
from .nudge import (
    nudge_reasoning_stepwise,
    nudge_with_regeneration,
    nudge_with_beam_search,
)
from .plan import (
    complete_reasoning_plan,
    auto_complete_missing_steps,
    iterative_refinement,
)

__all__ = [
    # Re-ranking
    "rerank_cots_with_jepa",
    "rerank_existing_cots",
    "rerank_with_ensembling",
    # Nudge
    "nudge_reasoning_stepwise",
    "nudge_with_regeneration",
    "nudge_with_beam_search",
    # Plan
    "complete_reasoning_plan",
    "auto_complete_missing_steps",
    "iterative_refinement",
]
