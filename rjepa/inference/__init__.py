"""
R-JEPA Inference Modes.

Four exploitation modes:
- Re-ranking: Choose best CoT among N candidates
- Nudge: Correct latents in real-time
- Plan: Complete missing steps
- Logit Guidance: Bias LLM logits with predicted latents
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
from .logit_guidance import (
    LogitGuidance,
    LogitGuidanceConfig,
    create_logit_guidance,
    guided_generation_step,
    generate_with_guidance,
)
from .logit_guidance_trainer import LogitGuidanceTrainer

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
    # Logit Guidance
    "LogitGuidance",
    "LogitGuidanceConfig",
    "create_logit_guidance",
    "guided_generation_step",
    "generate_with_guidance",
    "LogitGuidanceTrainer",
]
