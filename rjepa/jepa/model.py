# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# ADAPTED FOR R-JEPA: Complete Reasoning JEPA model
# Assembles StepTransformer (encoder) + StepPredictor + EMA target encoder

import copy
import torch
import torch.nn as nn

from rjepa.jepa.step_transformer import (
    StepTransformer,
    step_transformer_tiny,
    step_transformer_small,
    step_transformer_base,
    step_transformer_large,
    step_transformer_huge,
)
from rjepa.jepa.step_predictor import StepPredictor, step_predictor
from rjepa.jepa.multiblock1d import MaskCollator


class ReasoningJEPA(nn.Module):
    """Reasoning JEPA - Complete model"""

    def __init__(
        self,
        input_dim=4096,
        max_seq_len=512,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        predictor_embed_dim=384,
        predictor_depth=6,
        predictor_num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        ema_momentum=0.996,
        loss_exp=1.0,
        reg_coeff=0.01,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.encoder_embed_dim = encoder_embed_dim
        self.ema_momentum = ema_momentum
        self.loss_exp = loss_exp
        self.reg_coeff = reg_coeff

        self.context_encoder = StepTransformer(
            input_dim=input_dim,
            max_seq_len=max_seq_len,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_std=init_std,
        )

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = StepPredictor(
            max_seq_len=max_seq_len,
            encoder_dim=encoder_embed_dim,
            predictor_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_std=init_std,
            use_mask_tokens=use_mask_tokens,
            num_mask_tokens=num_mask_tokens,
            zero_init_mask_tokens=zero_init_mask_tokens,
        )

    def forward(self, x, masks_context, masks_target, mask_index=0, compute_loss=False):
        """
        Forward pass through Reasoning JEPA.

        Args:
            x: [B, S, D_in] input latent sequence
            masks_context: List of [B, K_c] context mask indices
            masks_target: List of [B, K_t] target mask indices
            mask_index: Which mask token to use (for predictor)
            compute_loss: Whether to compute JEPA loss (for training)

        Returns:
            Dict with keys:
                - z_context: [B*M_c, K_c, D] context encodings
                - z_target: [B*M_t, K_t, D] target encodings (detached)
                - z_pred: [B*M_t, K_t, D] predicted targets
                - loss: scalar (if compute_loss=True)
                - recon_loss: scalar (if compute_loss=True)
                - var_reg_loss: scalar (if compute_loss=True)
        """
        B, S, D_in = x.shape

        # Encode context (trainable)
        z_context = self.context_encoder(x, masks=masks_context)

        # Encode targets with EMA (no gradients)
        with torch.no_grad():
            z_target = self.target_encoder(x, masks=masks_target)
            # Layer norm like V-JEPA (normalize over feature dimension)
            z_target = torch.nn.functional.layer_norm(z_target, (z_target.size(-1),))

        # Project input for predictor (diffusion mode needs encoder_dim features, not input_dim)
        # This maps from input_dim (4096) to encoder_dim (768)
        x_proj = self.context_encoder.input_proj(x)  # [B, S, encoder_embed_dim]

        # Predict targets from context
        z_pred = self.predictor(
            ctxt=z_context,
            tgt=x_proj,  # Use projected features, not raw input
            masks_ctxt=masks_context,
            masks_tgt=masks_target,
            mask_index=mask_index,
        )

        result = {
            'z_context': z_context,
            'z_target': z_target.detach(),
            'z_pred': z_pred
        }

        if compute_loss:
            # V-JEPA loss computation (adapted from legacy-vjepa/app/vjepa/train.py:440-459)

            # 1. Reconstruction loss: |pred - target|^loss_exp / loss_exp
            #    This is a generalization of L1 (loss_exp=1) and L2 (loss_exp=2)
            loss_jepa = torch.mean(torch.abs(z_pred - z_target) ** self.loss_exp) / self.loss_exp

            # 2. Variance regularization: encourage predictions to have variance ~1
            #    Compute std dev across batch/sequence dimensions
            pstd_z = torch.sqrt(z_pred.var(dim=(0, 1)) + 0.0001)  # [D]
            loss_reg = torch.mean(torch.nn.functional.relu(1.0 - pstd_z))

            # 3. Total loss
            total_loss = loss_jepa + self.reg_coeff * loss_reg

            result.update({
                'loss': total_loss,
                'recon_loss': loss_jepa,
                'var_reg_loss': loss_reg,
            })

        return result

    @torch.no_grad()
    def update_target_encoder(self, momentum=None):
        if momentum is None:
            momentum = self.ema_momentum
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def get_num_layers(self):
        return self.context_encoder.get_num_layers()


def reasoning_jepa_base(input_dim=4096, max_seq_len=512, **kwargs):
    return ReasoningJEPA(
        input_dim=input_dim,
        max_seq_len=max_seq_len,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        predictor_embed_dim=384,
        predictor_depth=6,
        predictor_num_heads=12,
        **kwargs
    )

RJEPA_CONFIGS = {'base': reasoning_jepa_base}
