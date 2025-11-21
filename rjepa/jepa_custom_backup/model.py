"""
Reasoning-JEPA Model.

Complete world model architecture:
- Context Encoder (online, trained)
- Target Encoder (EMA, momentum update)
- Predictor (predicts target representations from context)

Philosophy: This is a world model of reasoning. It learns the stable
relationships between reasoning steps in latent space, enabling it to
predict what comes next conceptually (not token-by-token).
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, Optional
import copy

from .encoder import ReasoningEncoder
from .predictor import ReasoningPredictor
from .losses import JEPALoss

logger = logging.getLogger(__name__)


class ReasoningJEPA(nn.Module):
    """
    Reasoning Joint Embedding Predictive Architecture.

    Components:
    1. Context Encoder (online): Encodes visible reasoning steps
    2. Target Encoder (EMA): Encodes all steps with momentum averaging
    3. Predictor: Predicts target latents from context

    Training:
    - Context encoder + predictor are trained with gradients
    - Target encoder is updated via EMA (no gradients)
    - Loss: Predict target encoder outputs from context + predictor
    """

    def __init__(
        self,
        dim: int = 1024,
        depth_encoder: int = 12,
        depth_predictor: int = 8,
        num_heads: int = 16,
        predictor_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_steps: int = 512,
        domain_embed_dim: int = 64,
        num_domains: int = 50,
        ema_momentum: float = 0.996,
        loss_config: Optional[Dict] = None,
    ):
        """
        Initialize R-JEPA.

        Args:
            dim: Embedding dimension
            depth_encoder: Encoder depth (number of Transformer blocks)
            depth_predictor: Predictor depth
            num_heads: Number of attention heads
            predictor_dim: Predictor internal dimension (None = same as dim)
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout probability
            max_steps: Maximum sequence length
            domain_embed_dim: Domain embedding dimension
            num_domains: Number of domain classes
            ema_momentum: EMA momentum for target encoder (0.996-0.999)
            loss_config: Optional loss configuration dict
        """
        super().__init__()

        self.dim = dim
        self.ema_momentum = ema_momentum

        if predictor_dim is None:
            predictor_dim = dim

        # Context Encoder (online, trained with gradients)
        self.context_encoder = ReasoningEncoder(
            dim=dim,
            depth=depth_encoder,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_steps=max_steps,
            domain_embed_dim=domain_embed_dim,
            num_domains=num_domains,
        )

        # Target Encoder (EMA, no gradients)
        self.target_encoder = copy.deepcopy(self.context_encoder)

        # Freeze target encoder (no gradients)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor
        self.predictor = ReasoningPredictor(
            dim=dim,
            predictor_dim=predictor_dim,
            depth=depth_predictor,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Loss
        loss_config = loss_config or {}
        self.criterion = JEPALoss(**loss_config)

        logger.info(
            f"ReasoningJEPA initialized: "
            f"dim={dim}, encoder_depth={depth_encoder}, "
            f"predictor_depth={depth_predictor}, heads={num_heads}, "
            f"ema_momentum={ema_momentum}"
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """
        Update target encoder via EMA.

        Target encoder parameters ← momentum * target + (1-momentum) * context
        """
        for param_online, param_target in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_target.data.mul_(self.ema_momentum).add_(
                param_online.data, alpha=1 - self.ema_momentum
            )

    def forward(
        self,
        latents: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            latents: [B, S, D] pre-extracted latent representations
            context_mask: [B, S] boolean (True = context, False = target)
            target_mask: [B, S] boolean (True = target, False = context)
            domain_ids: Optional [B] domain IDs
            compute_loss: Whether to compute loss (False for inference)

        Returns:
            Dict with keys:
                - pred: [B, S, D] predictions
                - target: [B, S, D] targets (if compute_loss)
                - loss: scalar (if compute_loss)
                - ... (other loss components)
        """
        batch_size, seq_len, dim = latents.shape

        # Encode context (online encoder)
        context_encoded = self.context_encoder(
            latents,
            domain_ids=domain_ids,
            mask=context_mask,
        )  # [B, S, D]

        # Predict targets
        pred = self.predictor(
            context_latents=context_encoded,
            context_mask=context_mask,
            target_mask=target_mask,
        )  # [B, S, D]

        outputs = {
            "pred": pred,
        }

        if compute_loss:
            # Encode targets (EMA encoder, no gradients)
            with torch.no_grad():
                target_encoded = self.target_encoder(
                    latents,
                    domain_ids=domain_ids,
                    mask=None,  # Target encoder sees all steps
                )  # [B, S, D]

            # Compute loss
            losses = self.criterion(
                pred=pred,
                target=target_encoded,
                target_mask=target_mask,
            )

            outputs.update(losses)
            outputs["target"] = target_encoded

        return outputs

    def get_jepa_score(
        self,
        latents: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute JEPA score (lower = better coherence).

        This is the key metric for re-ranking, nudging, etc.

        Args:
            latents: [B, S, D] latents
            context_mask: [B, S] context mask
            target_mask: [B, S] target mask
            domain_ids: Optional [B] domain IDs

        Returns:
            [B] tensor of JEPA scores (reconstruction loss per sample)
        """
        with torch.no_grad():
            outputs = self.forward(
                latents,
                context_mask,
                target_mask,
                domain_ids,
                compute_loss=True,
            )

            # Compute per-sample loss
            pred = outputs["pred"]  # [B, S, D]
            target = outputs["target"]  # [B, S, D]

            # L1 loss per sample
            loss_per_step = (pred - target).abs().mean(dim=-1)  # [B, S]
            loss_per_sample = (loss_per_step * target_mask.float()).sum(dim=1) / (
                target_mask.float().sum(dim=1) + 1e-8
            )  # [B]

            return loss_per_sample


def create_rjepa_model(config: Dict) -> ReasoningJEPA:
    """
    Create R-JEPA model from config dict.

    Args:
        config: Configuration dict

    Returns:
        ReasoningJEPA model
    """
    model = ReasoningJEPA(
        dim=config.get("dim", 1024),
        depth_encoder=config.get("depth_encoder", 12),
        depth_predictor=config.get("depth_predictor", 8),
        num_heads=config.get("num_heads", 16),
        predictor_dim=config.get("predictor_dim", None),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        dropout=config.get("dropout", 0.0),
        max_steps=config.get("max_steps", 512),
        domain_embed_dim=config.get("domain_embed_dim", 64),
        num_domains=config.get("num_domains", 50),
        ema_momentum=config.get("ema_momentum", 0.996),
        loss_config=config.get("loss", {}),
    )

    return model
