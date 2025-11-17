"""
Reasoning Predictor for R-JEPA.

Predicts masked reasoning step latents from context.
This is where the "world model" magic happens: predicting what comes next
in the reasoning chain based on what we've seen so far.
"""
import logging
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class PredictorBlock(nn.Module):
    """Transformer block for predictor (similar to encoder but separate)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        Initialize predictor block.

        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout probability
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, S, D] input tensor

        Returns:
            [B, S, D] output tensor
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class ReasoningPredictor(nn.Module):
    """
    Reasoning Predictor.

    Takes encoded context steps and predicts the masked target steps.

    Architecture:
    - Input: context latents [B, S_ctx, D] + mask tokens for targets [B, S_tgt, D]
    - Transformer: Process combined sequence
    - Output: Predictions for target positions [B, S_tgt, D]

    Philosophy: This learns the "physics" of reasoning - how steps logically
    flow from each other. It's not predicting tokens, but conceptual states.
    """

    def __init__(
        self,
        dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        Initialize predictor.

        Args:
            dim: Input embedding dimension (from encoder)
            predictor_dim: Internal predictor dimension
            depth: Number of Transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout probability
        """
        super().__init__()

        self.dim = dim
        self.predictor_dim = predictor_dim
        self.depth = depth

        # Project input to predictor dimension
        if dim != predictor_dim:
            self.input_proj = nn.Linear(dim, predictor_dim)
        else:
            self.input_proj = nn.Identity()

        # Learnable mask tokens (for target positions)
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim))

        # Predictor Transformer blocks
        self.blocks = nn.ModuleList(
            [
                PredictorBlock(
                    dim=predictor_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # Final norm
        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to original dimension
        if predictor_dim != dim:
            self.output_proj = nn.Linear(predictor_dim, dim)
        else:
            self.output_proj = nn.Identity()

        logger.info(
            f"ReasoningPredictor initialized: "
            f"dim={dim}, predictor_dim={predictor_dim}, "
            f"depth={depth}, heads={num_heads}"
        )

    def forward(
        self,
        context_latents: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict masked target latents from context.

        Args:
            context_latents: [B, S, D] all latents (context + target positions)
            context_mask: [B, S] boolean (True = context, False = target)
            target_mask: [B, S] boolean (True = target, False = context)

        Returns:
            [B, S, D] predictions for ALL positions (but only target positions matter)
        """
        batch_size, seq_len, _ = context_latents.shape

        # Project to predictor dimension
        x = self.input_proj(context_latents)  # [B, S, predictor_dim]

        # Replace target positions with mask tokens
        mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)  # [B, S, predictor_dim]

        # Use context where context_mask=True, mask_token where False
        x = torch.where(
            context_mask.unsqueeze(-1).expand_as(x),
            x,
            mask_tokens,
        )

        # Apply predictor blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        # Project back to original dimension
        x = self.output_proj(x)  # [B, S, D]

        return x
