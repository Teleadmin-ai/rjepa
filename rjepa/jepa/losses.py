"""
Loss functions for R-JEPA.

Multi-component loss:
1. L1 loss on predicted vs target latents (main reconstruction)
2. Variance regularization (prevent collapse)
3. (Optional) Contrastive loss (discriminative learning)
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class JEPALoss(nn.Module):
    """
    Combined loss for R-JEPA training.

    Components:
    - L1 reconstruction loss (main)
    - Variance regularization (prevent representation collapse)
    - Optional contrastive loss (make predictions discriminative)
    """

    def __init__(
        self,
        loss_type: str = "l1",
        var_reg_weight: float = 0.01,
        var_reg_target: float = 1.0,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.07,
    ):
        """
        Initialize JEPA loss.

        Args:
            loss_type: "l1" or "l2" (L1 is more robust, recommended)
            var_reg_weight: Weight for variance regularization
            var_reg_target: Target variance (1.0 = unit variance)
            contrastive_weight: Weight for contrastive loss (0.0 = disabled)
            contrastive_temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.loss_type = loss_type
        self.var_reg_weight = var_reg_weight
        self.var_reg_target = var_reg_target
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature

        logger.info(
            f"JEPALoss initialized: "
            f"type={loss_type}, var_reg={var_reg_weight}, "
            f"contrastive={contrastive_weight}"
        )

    def reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss (L1 or L2).

        Args:
            pred: [B, S, D] predicted latents
            target: [B, S, D] target latents
            mask: Optional [B, S] boolean mask (True = compute loss)

        Returns:
            Scalar loss
        """
        if self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")  # [B, S, D]
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred, target, reduction="none")  # [B, S, D]
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Average over dimension
        loss = loss.mean(dim=-1)  # [B, S]

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / (mask.sum() + 1e-8)
        else:
            return loss.mean()

    def variance_regularization(
        self,
        pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Variance regularization to prevent representation collapse.

        Encourages predictions to have variance close to target variance.

        Args:
            pred: [B, S, D] predicted latents
            mask: Optional [B, S] boolean mask

        Returns:
            Scalar loss
        """
        # Compute variance along batch dimension
        if mask is not None:
            # Masked variance
            mask_expanded = mask.unsqueeze(-1).float()  # [B, S, 1]
            count = mask_expanded.sum(dim=(0, 1)) + 1e-8  # [D]

            mean = (pred * mask_expanded).sum(dim=(0, 1)) / count  # [D]
            var = ((pred - mean) ** 2 * mask_expanded).sum(dim=(0, 1)) / count  # [D]
        else:
            # Unmasked variance
            var = pred.var(dim=(0, 1))  # [D]

        # Loss = mean squared deviation from target variance
        loss = ((var.sqrt() - self.var_reg_target) ** 2).mean()

        return loss

    def contrastive_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive loss (InfoNCE-style).

        For each target step, treat its prediction as positive,
        and predictions of other steps in the batch as negatives.

        This makes the model more discriminative: not just predicting
        "something plausible", but the ACTUAL next step.

        Args:
            pred: [B, S, D] predicted latents
            target: [B, S, D] target latents
            mask: [B, S] boolean mask (True = target step)

        Returns:
            Scalar loss
        """
        # Flatten to [N, D] where N = number of target steps
        pred_flat = pred[mask]  # [N, D]
        target_flat = target[mask]  # [N, D]

        if pred_flat.size(0) == 0:
            return torch.tensor(0.0, device=pred.device)

        # Normalize
        pred_norm = F.normalize(pred_flat, dim=-1)  # [N, D]
        target_norm = F.normalize(target_flat, dim=-1)  # [N, D]

        # Similarity matrix
        sim_matrix = torch.matmul(pred_norm, target_norm.T)  # [N, N]
        sim_matrix = sim_matrix / self.contrastive_temperature

        # Positive pairs are on diagonal
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            pred: [B, S, D] predicted latents
            target: [B, S, D] target latents (from EMA encoder)
            target_mask: [B, S] boolean mask (True = target position)

        Returns:
            Dict with keys:
                - loss: total loss
                - recon_loss: reconstruction component
                - var_reg_loss: variance regularization component
                - contrastive_loss: contrastive component (if enabled)
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(pred, target, mask=target_mask)

        # Variance regularization
        var_reg_loss = self.variance_regularization(pred, mask=target_mask)

        # Total loss
        total_loss = recon_loss + self.var_reg_weight * var_reg_loss

        losses = {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "var_reg_loss": var_reg_loss,
        }

        # Contrastive loss (optional)
        if self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss(pred, target, target_mask)
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            losses["loss"] = total_loss
            losses["contrastive_loss"] = contrastive_loss

        return losses
