"""
Logit Guidance: Bias LLM logits with predicted latents.

Philosophy:
- R-JEPA predicts the next latent vector ĥ_next
- We project this latent to vocabulary space: ĥ → logit_bias
- During generation, add bias to LLM logits: logits_final = logits_llm + α * logit_bias
- This "pulls" generation toward tokens that would produce latents close to ĥ

Advantage over nudge mode:
- Works without direct access to LLM hidden states
- Compatible with API-based LLMs (OpenAI, Anthropic, etc.)
- Less invasive (doesn't modify internal representations)

Inspired by:
- Classifier-free guidance (diffusion models)
- Contrastive decoding (LLM literature)
"""
import logging
from typing import Optional, Dict, List
import torch
import torch.nn as nn
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LogitGuidanceConfig:
    """Configuration for LogitGuidance."""

    latent_dim: int = 4096  # Input latent dimension (R-JEPA output)
    vocab_size: int = 151936  # LLM vocabulary size
    hidden_dim: int = 2048  # MLP hidden dimension
    dropout: float = 0.1
    alpha: float = 0.3  # Guidance strength (0 = no guidance, 1 = full guidance)
    temperature: float = 1.0  # Temperature for logit bias


class LogitGuidance(nn.Module):
    """
    Projects latent vectors to vocabulary logit biases.

    Architecture:
    - Latent [latent_dim] → MLP [hidden_dim] → Logit bias [vocab_size]
    - Small MLP (2-3 layers) to keep it lightweight
    - Trained jointly with R-JEPA or separately

    Usage:
    1. R-JEPA predicts ĥ_next for next step
    2. LogitGuidance projects ĥ_next → logit_bias
    3. Add bias to LLM logits: logits_final = logits_llm + α * logit_bias
    4. Sample next token from logits_final
    """

    def __init__(self, config: LogitGuidanceConfig):
        """
        Initialize LogitGuidance.

        Args:
            config: LogitGuidanceConfig
        """
        super().__init__()

        self.config = config

        # MLP: latent → hidden → vocab
        self.mlp = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.vocab_size),
        )

        # Initialize with small weights (don't want to overwhelm LLM logits)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        logger.info(
            f"LogitGuidance initialized: latent_dim={config.latent_dim}, "
            f"vocab_size={config.vocab_size}, alpha={config.alpha}"
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Project latent to logit bias.

        Args:
            latent: [B, latent_dim] or [latent_dim] tensor

        Returns:
            logit_bias: [B, vocab_size] or [vocab_size] tensor
        """
        # Ensure batch dimension
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)  # [1, latent_dim]

        # Project to vocab space
        logit_bias = self.mlp(latent)  # [B, vocab_size]

        # Apply temperature to bias (soften or sharpen)
        logit_bias = logit_bias / self.config.temperature

        return logit_bias

    def apply_guidance(
        self,
        llm_logits: torch.Tensor,
        latent: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply logit guidance to LLM logits.

        Args:
            llm_logits: [B, vocab_size] logits from LLM
            latent: [B, latent_dim] predicted latent from R-JEPA
            alpha: Guidance strength (overrides config.alpha if provided)

        Returns:
            guided_logits: [B, vocab_size] biased logits
        """
        if alpha is None:
            alpha = self.config.alpha

        # Compute logit bias from latent
        logit_bias = self.forward(latent)  # [B, vocab_size]

        # Apply guidance: logits_final = logits_llm + α * logit_bias
        guided_logits = llm_logits + alpha * logit_bias

        return guided_logits


def guided_generation_step(
    llm_model,
    rjepa_model,
    logit_guidance: LogitGuidance,
    input_ids: torch.Tensor,
    context_latents: torch.Tensor,
    alpha: float = 0.3,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> Dict[str, torch.Tensor]:
    """
    Single generation step with logit guidance.

    Workflow:
    1. LLM forward pass → logits_llm
    2. R-JEPA predicts next latent: ĥ_next = predict(context_latents)
    3. LogitGuidance projects: logit_bias = guidance(ĥ_next)
    4. Apply guidance: logits_final = logits_llm + α * logit_bias
    5. Sample next token from logits_final

    Args:
        llm_model: LLM model (for generation)
        rjepa_model: R-JEPA model (for latent prediction)
        logit_guidance: LogitGuidance module
        input_ids: [B, S] current sequence
        context_latents: [B, T, latent_dim] latent history
        alpha: Guidance strength
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        {
          "next_token": [B, 1] next token,
          "logits_llm": [B, vocab_size] LLM logits,
          "logits_guided": [B, vocab_size] guided logits,
          "predicted_latent": [B, latent_dim] predicted next latent
        }
    """
    batch_size = input_ids.shape[0]

    # 1. LLM forward pass
    with torch.no_grad():
        llm_outputs = llm_model(input_ids, output_hidden_states=False)
        logits_llm = llm_outputs.logits[:, -1, :]  # [B, vocab_size]

    # 2. R-JEPA predicts next latent
    with torch.no_grad():
        # Predict next latent based on context
        # Assuming rjepa_model has a predict_next() method
        predicted_latent = rjepa_model.predict_next(context_latents)  # [B, latent_dim]

    # 3. Apply logit guidance
    logits_guided = logit_guidance.apply_guidance(
        llm_logits=logits_llm, latent=predicted_latent, alpha=alpha
    )

    # 4. Sample next token
    # Apply temperature
    logits_final = logits_guided / temperature

    # Top-p (nucleus) sampling
    sorted_logits, sorted_indices = torch.sort(logits_final, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1
    )

    # Remove tokens with cumulative probability > top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    # Mask logits
    for i in range(batch_size):
        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
        logits_final[i, indices_to_remove] = float("-inf")

    # Sample
    probs = torch.softmax(logits_final, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

    return {
        "next_token": next_token,
        "logits_llm": logits_llm,
        "logits_guided": logits_guided,
        "predicted_latent": predicted_latent,
    }


def generate_with_guidance(
    llm_model,
    rjepa_model,
    logit_guidance: LogitGuidance,
    llm_adapter,
    prompt: str,
    max_new_tokens: int = 256,
    alpha: float = 0.3,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, any]:
    """
    Full autoregressive generation with logit guidance.

    Args:
        llm_model: LLM model
        rjepa_model: R-JEPA model
        logit_guidance: LogitGuidance module
        llm_adapter: LLMAdapter (for tokenization and latent extraction)
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        alpha: Guidance strength
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        {
          "text": str,
          "tokens": [1, T] tensor,
          "steps": List[str],
          "latents": [T, latent_dim] tensor,
          "guidance_history": List[Dict]  # Per-step guidance info
        }
    """
    # Tokenize prompt
    input_ids = llm_adapter.tokenizer.encode(
        prompt, return_tensors="pt"
    ).to(llm_model.device)

    # Extract initial latents from prompt
    with torch.no_grad():
        prompt_outputs = llm_model(input_ids, output_hidden_states=True)
        # Extract from layer -2
        initial_latent = prompt_outputs.hidden_states[-2][:, -1, :]  # [1, latent_dim]

    context_latents = initial_latent.unsqueeze(1)  # [1, 1, latent_dim]
    guidance_history = []

    # Autoregressive generation
    for step in range(max_new_tokens):
        # Single step with guidance
        step_result = guided_generation_step(
            llm_model=llm_model,
            rjepa_model=rjepa_model,
            logit_guidance=logit_guidance,
            input_ids=input_ids,
            context_latents=context_latents,
            alpha=alpha,
            temperature=temperature,
            top_p=top_p,
        )

        # Append next token
        next_token = step_result["next_token"]
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Update context latents
        # Extract latent for the new token
        with torch.no_grad():
            new_outputs = llm_model(input_ids, output_hidden_states=True)
            new_latent = new_outputs.hidden_states[-2][:, -1, :]  # [1, latent_dim]

        context_latents = torch.cat(
            [context_latents, new_latent.unsqueeze(1)], dim=1
        )

        # Store guidance history
        guidance_history.append(
            {
                "step": step,
                "token_id": next_token.item(),
                "logit_bias_max": (step_result["logits_guided"] - step_result["logits_llm"]).max().item(),
                "logit_bias_mean": (step_result["logits_guided"] - step_result["logits_llm"]).mean().item(),
            }
        )

        # Check for EOS
        if next_token.item() == llm_adapter.tokenizer.eos_token_id:
            break

    # Decode to text
    text = llm_adapter.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Segment into steps (if structured)
    steps, step_boundaries = llm_adapter._segment_into_steps(text, input_ids[0])

    # Extract final latents
    latents = context_latents.squeeze(0)  # [T, latent_dim]

    return {
        "text": text,
        "tokens": input_ids,
        "steps": steps,
        "latents": latents,
        "guidance_history": guidance_history,
    }


def create_logit_guidance(
    latent_dim: int = 4096,
    vocab_size: int = 151936,
    hidden_dim: int = 2048,
    alpha: float = 0.3,
    **kwargs,
) -> LogitGuidance:
    """
    Factory function to create LogitGuidance.

    Args:
        latent_dim: Latent dimension
        vocab_size: Vocabulary size
        hidden_dim: MLP hidden dimension
        alpha: Guidance strength
        **kwargs: Additional config parameters

    Returns:
        LogitGuidance instance
    """
    config = LogitGuidanceConfig(
        latent_dim=latent_dim,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        alpha=alpha,
        **kwargs,
    )

    guidance = LogitGuidance(config)

    num_params = sum(p.numel() for p in guidance.parameters())
    logger.info(f"Created LogitGuidance with {num_params:,} parameters")

    return guidance
