"""
Reasoning Encoder for R-JEPA.

Transformer-based encoder that processes visible reasoning steps (context).
Inspired by V-JEPA ViT encoder, adapted for 1D sequences of reasoning steps.
"""
import logging
import torch
import torch.nn as nn
import math
from typing import Optional

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for step positions."""

    def __init__(self, dim: int, max_steps: int = 512):
        """
        Initialize positional encoding.

        Args:
            dim: Embedding dimension
            max_steps: Maximum number of steps
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_steps, dim)
        position = torch.arange(0, max_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Args:
            x: [B, S, D] tensor

        Returns:
            [B, S, D] tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class TransformerBlock(nn.Module):
    """Single Transformer block (multi-head attention + FFN)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        Initialize Transformer block.

        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, S, D] input tensor
            mask: Optional [B, S] boolean mask (True = keep, False = mask)

        Returns:
            [B, S, D] output tensor
        """
        # Self-attention with residual
        x_norm = self.norm1(x)

        # Create attention mask if provided
        attn_mask = None
        if mask is not None:
            # MultiheadAttention expects mask where True = IGNORE
            # So we invert our mask
            attn_mask = ~mask  # [B, S]

        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=attn_mask,
        )
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class ReasoningEncoder(nn.Module):
    """
    Reasoning Encoder (Context Encoder).

    Encodes visible reasoning steps into contextualized representations.
    This is the "perception" part of the world model.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_steps: int = 512,
        domain_embed_dim: int = 0,
        num_domains: int = 50,
    ):
        """
        Initialize encoder.

        Args:
            dim: Embedding dimension
            depth: Number of Transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout probability
            max_steps: Maximum sequence length
            domain_embed_dim: Domain embedding dimension (0 = no domain embedding)
            num_domains: Number of domain classes
        """
        super().__init__()

        self.dim = dim
        self.depth = depth

        # Positional encoding
        self.pos_encoding = PositionalEncoding(dim, max_steps)

        # Domain embedding (optional)
        self.domain_embed_dim = domain_embed_dim
        if domain_embed_dim > 0:
            self.domain_embed = nn.Embedding(num_domains, domain_embed_dim)
            # Project to match dim if needed
            if domain_embed_dim != dim:
                self.domain_proj = nn.Linear(domain_embed_dim, dim)
            else:
                self.domain_proj = nn.Identity()
        else:
            self.domain_embed = None
            self.domain_proj = None

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # Final norm
        self.norm = nn.LayerNorm(dim)

        logger.info(
            f"ReasoningEncoder initialized: "
            f"dim={dim}, depth={depth}, heads={num_heads}, "
            f"domain_embed_dim={domain_embed_dim}"
        )

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode reasoning steps.

        Args:
            x: [B, S, D] latent representations of steps
            domain_ids: Optional [B] domain IDs for embedding
            mask: Optional [B, S] boolean mask (True = visible, False = masked)

        Returns:
            [B, S, D] encoded representations
        """
        # Add positional encoding
        x = self.pos_encoding(x)

        # Add domain embedding if provided
        if self.domain_embed is not None and domain_ids is not None:
            domain_emb = self.domain_embed(domain_ids)  # [B, domain_embed_dim]
            domain_emb = self.domain_proj(domain_emb)  # [B, dim]
            # Add to all steps
            x = x + domain_emb.unsqueeze(1)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Final norm
        x = self.norm(x)

        return x
