# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# ADAPTED FOR R-JEPA: Vision Transformer → Step Transformer
# Adapts V-JEPA 2's encoder from 2D/3D patches to 1D reasoning step sequences

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


class StepTransformer(nn.Module):
    """
    Step Transformer - Adapted from V-JEPA's VisionTransformer for 1D sequences.

    Processes sequences of reasoning steps (already converted to latent vectors).
    Unlike VisionTransformer which processes raw images/videos, this processes
    pre-extracted latent representations from an LLM.

    Key differences from VisionTransformer:
    - No PatchEmbed (we work with pre-extracted latents)
    - 1D sinusoidal positional embeddings instead of 2D/3D
    - No interpolation needed (sequence length is dynamic but handled simply)
    - Input projection layer to map LLM hidden_size to encoder embed_dim
    """

    def __init__(
        self,
        input_dim=4096,          # LLM hidden size (e.g., Qwen3-8B = 4096)
        max_seq_len=512,         # Maximum sequence length (reasoning steps)
        embed_dim=768,           # Encoder embedding dimension
        depth=12,                # Number of transformer blocks
        num_heads=12,            # Number of attention heads
        mlp_ratio=4.0,           # MLP hidden dim ratio
        qkv_bias=True,           # Bias in QKV projections
        qk_scale=None,           # Manual scale for QK attention
        drop_rate=0.0,           # Dropout rate
        attn_drop_rate=0.0,      # Attention dropout rate
        drop_path_rate=0.0,      # Stochastic depth rate (V-JEPA 2)
        norm_layer=nn.LayerNorm,
        init_std=0.02,           # Weight initialization std
        out_layers=None,         # Which layers to output (for multi-scale)
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.max_seq_len = max_seq_len

        # Input projection: LLM hidden_size -> encoder embed_dim
        # This is analogous to PatchEmbed in VisionTransformer
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Positional embedding (1D sinusoidal, fixed)
        # We create for max_seq_len and slice as needed
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim),
            requires_grad=False
        )

        # Stochastic depth decay rule (V-JEPA 2)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks (V-JEPA 2 style with DropPath)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
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
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        self._init_pos_embed(self.pos_embed.data)
        self.init_std = init_std
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

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, masks=None):
        """
        Forward pass through Step Transformer.

        Args:
            x: [B, S, D_in] - Batch of step sequences (D_in = LLM hidden size)
            masks: Optional list of [B, K] tensors with indices of steps to keep

        Returns:
            [B, S', D_out] - Encoded step representations
            (S' = S if no masks, else sum of kept steps across masks)
        """

        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        B, S, D_in = x.shape

        # Project input latents to encoder dimension
        x = self.input_proj(x)  # [B, S, embed_dim]

        # Add positional embeddings (slice to sequence length)
        pos_embed = self.pos_embed[:, :S, :]  # [1, S, embed_dim]
        x = x + pos_embed

        # Apply masks if provided (keep only certain steps)
        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        # Forward through transformer blocks
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        # Return outputs
        if self.out_layers is not None:
            return outs

        if self.norm is not None:
            x = self.norm(x)

        return x


# Model size variants (analogous to V-JEPA's vit_tiny, vit_small, etc.)

def step_transformer_tiny(input_dim=4096, **kwargs):
    """Tiny Step Transformer (12M params)"""
    model = StepTransformer(
        input_dim=input_dim,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def step_transformer_small(input_dim=4096, **kwargs):
    """Small Step Transformer (22M params)"""
    model = StepTransformer(
        input_dim=input_dim,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def step_transformer_base(input_dim=4096, **kwargs):
    """Base Step Transformer (86M params) - RECOMMENDED for MVP"""
    model = StepTransformer(
        input_dim=input_dim,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def step_transformer_large(input_dim=4096, **kwargs):
    """Large Step Transformer (307M params)"""
    model = StepTransformer(
        input_dim=input_dim,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def step_transformer_huge(input_dim=4096, **kwargs):
    """Huge Step Transformer (632M params)"""
    model = StepTransformer(
        input_dim=input_dim,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


STEP_TRANSFORMER_EMBED_DIMS = {
    'step_transformer_tiny': 192,
    'step_transformer_small': 384,
    'step_transformer_base': 768,
    'step_transformer_large': 1024,
    'step_transformer_huge': 1280,
}
