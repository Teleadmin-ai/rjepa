# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# ADAPTED FOR R-JEPA: VisionTransformerPredictor → StepPredictor
# Adapts V-JEPA 2's predictor from 2D/3D patches to 1D reasoning step sequences

import math
from functools import partial

import torch
import torch.nn as nn

from rjepa.jepa.modules import Block
from rjepa.jepa.pos_embs import get_1d_sincos_pos_embed
from rjepa.jepa.vjepa_adapted.utils import apply_masks


def trunc_normal_(tensor, mean=0., std=1.):
    """
    Truncated normal initialization.

    CRITICAL FIX: Use PyTorch's built-in trunc_normal_ to avoid GIL issues.
    The custom implementation had threading problems on Windows.
    """
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2.0, b=2.0)


def repeat_interleave_batch(x, B, repeat):
    """
    Repeat tensors along batch dimension (from V-JEPA).

    Args:
        x: [B, N, D] tensor
        B: original batch size
        repeat: number of times to repeat

    Returns:
        [B*repeat, N, D] tensor with each batch element repeated
    """
    # CRITICAL FIX: Handle empty tensors from masking edge cases
    if x.numel() == 0:
        # Return empty tensor with correct shape
        return x

    N = x.shape[1]
    x = x.view(B, 1, N, -1)
    x = x.expand(B, repeat, N, -1)
    x = x.reshape(B * repeat, N, -1)
    return x


class StepPredictor(nn.Module):
    """
    Step Predictor - Adapted from V-JEPA's VisionTransformerPredictor for 1D sequences.

    Given context steps (visible), predicts target steps (masked).

    Architecture:
    1. Map context from encoder_dim → predictor_dim
    2. Initialize targets with mask tokens OR noisy versions of ground truth
    3. Add positional embeddings to both context and targets
    4. Concatenate and process through transformer blocks
    5. Project predictions back to encoder_dim

    This is the core of JEPA: predict latent representations of masked steps
    from visible context, NOT predict tokens (that would be autoregressive LM).
    """

    def __init__(
        self,
        max_seq_len=512,         # Maximum sequence length
        encoder_dim=768,         # Encoder output dimension
        predictor_dim=384,       # Predictor internal dimension (typically smaller)
        depth=6,                 # Number of predictor transformer blocks
        num_heads=12,            # Number of attention heads
        mlp_ratio=4.0,           # MLP hidden dim ratio
        qkv_bias=True,           # Bias in QKV projections
        qk_scale=None,           # Manual scale for QK attention
        drop_rate=0.0,           # Dropout rate
        attn_drop_rate=0.0,      # Attention dropout rate
        drop_path_rate=0.0,      # Stochastic depth rate (V-JEPA 2)
        norm_layer=nn.LayerNorm,
        init_std=0.02,           # Weight initialization std
        use_mask_tokens=False,   # Use learnable mask tokens vs diffusion
        num_mask_tokens=2,       # Number of different mask tokens (if used)
        zero_init_mask_tokens=True,  # Initialize mask tokens to zero
        **kwargs
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.max_seq_len = max_seq_len

        # Map encoder output to predictor dimension
        self.predictor_embed = nn.Linear(encoder_dim, predictor_dim, bias=True)

        # Mask tokens (learnable tokens for masked positions)
        # Alternative to diffusion noise
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_dim))
                for i in range(num_mask_tokens)
            ])

        # Positional embedding (1D sinusoidal, fixed)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, predictor_dim),
            requires_grad=False
        )

        # Stochastic depth decay rule (V-JEPA 2)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks (V-JEPA 2 style with DropPath)
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],    # V-JEPA 2: stochastic depth
                norm_layer=norm_layer,
                grid_size=None,      # Not used for 1D
                grid_depth=None,     # Not used for 1D
            )
            for i in range(depth)
        ])

        # Normalize & project back to encoder dimension
        self.predictor_norm = norm_layer(predictor_dim)
        self.predictor_proj = nn.Linear(predictor_dim, encoder_dim, bias=True)

        # Initialize weights
        self._init_pos_embed(self.predictor_pos_embed.data)
        self.init_std = init_std
        if not zero_init_mask_tokens and self.mask_tokens is not None:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        """Initialize 1D sinusoidal positional embeddings"""
        embed_dim = pos_embed.size(-1)
        max_len = pos_embed.size(1)

        sincos = get_1d_sincos_pos_embed(embed_dim, max_len)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        """Initialize weights (from V-JEPA)"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        """Rescale residual blocks (from V-JEPA)"""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def diffusion(self, x, noise_beta=(0.5, 1.0), steps=1000):
        """
        Apply forward diffusion to target tokens (from V-JEPA).

        Adds noise to target latents to make prediction task harder and
        prevent trivial copying. Uses DDPM-style noise schedule.

        Args:
            x: [B, N, D] target latent tokens
            noise_beta: (min, max) noise schedule bounds
            steps: number of diffusion steps

        Returns:
            Noisy version of x
        """
        # Prepare diffusion noise schedule
        b1, b2 = noise_beta
        beta_scheduler = [b1 + i*(b2-b1)/steps for i in range(steps)]
        alpha_scheduler = []
        _alpha = 1.0
        for _beta in beta_scheduler:
            _alpha *= 1. - _beta
            alpha_scheduler.append(_alpha)

        # Sample diffusion time step (random for each batch element)
        T = torch.randint(0, steps, (len(x),), device=x.device)
        alpha = torch.tensor(alpha_scheduler, device=x.device)[T].unsqueeze(-1).unsqueeze(-1)

        # Normalize features and apply noise
        # x_t = sqrt(alpha) * x + sqrt(1 - alpha) * noise
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))
        x = alpha**0.5 * x + (1. - alpha)**0.5 * torch.randn_like(x)
        return x

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt, mask_index=1):
        """
        Predict target step latents from context step latents.

        Args:
            ctxt: [B*M_c, N_ctxt, encoder_dim] - Context tokens from encoder
            tgt: [B, max_seq_len, encoder_dim] - Target tokens (ground truth, for diffusion)
                 Can be None if using mask_tokens
            masks_ctxt: List of [B, K_c] tensors - Indices of context steps
            masks_tgt: List of [B, K_t] tensors - Indices of target steps to predict
            mask_index: Which mask token to use (if use_mask_tokens=True)

        Returns:
            [B*M_t, N_tgt, encoder_dim] - Predicted target latents

        Note: M_c = len(masks_ctxt), M_t = len(masks_tgt)
              In standard JEPA, M_c = M_t (aligned 1:1)
        """

        assert (masks_ctxt is not None) and (masks_tgt is not None), \
            'Cannot run predictor without mask indices'

        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]

        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        # Batch size (original, before masking)
        B = len(ctxt) // len(masks_ctxt)

        # Map context tokens to predictor dimension
        x = self.predictor_embed(ctxt)  # [B*M_c, N_ctxt, predictor_dim]
        _, N_ctxt, D = x.shape

        # Add positional embeddings to context tokens
        if self.predictor_pos_embed is not None:
            ctxt_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)  # [B, max_seq_len, D]
            x += apply_masks(ctxt_pos_embed, masks_ctxt)  # Keep only context positions

        # Initialize target tokens
        # Option 1: Learnable mask tokens (simple, no ground truth needed)
        # Option 2: Diffusion (add noise to ground truth targets)
        if self.mask_tokens is None:
            # Diffusion mode: map targets to predictor dim and add noise
            pred_tokens = self.predictor_embed(tgt)  # [B, max_seq_len, predictor_dim]
            pred_tokens = self.diffusion(pred_tokens)
            pred_tokens = apply_masks(pred_tokens, masks_tgt)  # Keep only target positions
        else:
            # Mask token mode: use learnable tokens
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]  # [1, 1, predictor_dim]
            pred_tokens = pred_tokens.repeat(B, self.max_seq_len, 1)  # [B, max_seq_len, D]
            pred_tokens = apply_masks(pred_tokens, masks_tgt)  # [B*M_t, N_tgt, D]

        # Add positional embeddings to target tokens
        if self.predictor_pos_embed is not None:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)  # [B, max_seq_len, D]
            pos_embs = apply_masks(pos_embs, masks_tgt)  # [B*M_t, N_tgt, D]
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt))
            pred_tokens += pos_embs

        # Concatenate context & target tokens
        # Context: [B*M_c, N_ctxt, D]
        # Targets: [B*M_t, N_tgt, D]
        # → Repeat context M_t times: [B*M_c*M_t, N_ctxt, D]
        # → Concatenate: [B*M_c*M_t, N_ctxt+N_tgt, D]
        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)  # [B*M_t*M_c, N_ctxt+N_tgt, D]

        # Combine context and target masks for attention
        # IMPORTANT: This assumes masks_ctxt and masks_tgt are aligned 1:1
        # (works with standard JEPA masking strategies)
        masks_ctxt_cat = torch.cat(masks_ctxt, dim=0)  # [B*M_c, K_c]
        masks_tgt_cat = torch.cat(masks_tgt, dim=0)    # [B*M_t, K_t]
        masks = torch.cat([masks_ctxt_cat, masks_tgt_cat], dim=1)  # [B*M, K_c+K_t]

        # Forward through predictor transformer blocks
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # Extract only the predicted target tokens
        # x: [B*M, N_ctxt+N_tgt, D] → x[:, N_ctxt:]: [B*M, N_tgt, D]
        x = x[:, N_ctxt:]

        # Project back to encoder dimension
        x = self.predictor_proj(x)  # [B*M, N_tgt, encoder_dim]

        return x


def step_predictor(
    max_seq_len=512,
    encoder_dim=768,
    predictor_dim=384,
    depth=6,
    **kwargs
):
    """
    Standard Step Predictor factory.

    Default config:
    - predictor_dim = encoder_dim / 2 (more efficient)
    - depth = 6 (shallower than encoder, typically 12)
    - use_mask_tokens = False (diffusion by default)
    """
    model = StepPredictor(
        max_seq_len=max_seq_len,
        encoder_dim=encoder_dim,
        predictor_dim=predictor_dim,
        depth=depth,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
