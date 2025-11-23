# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# ADAPTED FOR R-JEPA: 1D sinusoidal positional embeddings for reasoning steps
# V-JEPA 2 already includes get_1d_sincos_pos_embed, we use the same implementation.

import numpy as np


def get_1d_sincos_pos_embed(embed_dim: int, seq_len: int, cls_token: bool = False):
    """
    Generate 1D sinusoidal positional embeddings for sequence of steps.

    This is the V-JEPA 2 implementation for 1D sequences.

    Args:
        embed_dim: output dimension for each position
        seq_len: int of the sequence length (number of reasoning steps)
        cls_token: if True, prepend a zero vector for CLS token

    Returns:
        pos_embed: [seq_len, embed_dim] or [1 + seq_len, embed_dim] if cls_token
    """
    grid = np.arange(seq_len, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray):
    """
    Generate positional embedding from explicit positions.

    Args:
        embed_dim: output dimension for each position
        pos: array of positions [M,]

    Returns:
        pos_embed: [M, embed_dim]
    """
    assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos encoding"

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
