# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# ADAPTED FOR R-JEPA: 1D sinusoidal positional embeddings for reasoning steps

import math
import numpy as np
import torch


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Generate 1D sinusoidal positional embeddings for sequence of steps.

    Adapted from V-JEPA's get_2d_sincos_pos_embed for 1D sequences.

    Args:
        embed_dim: embedding dimension
        length: number of steps in sequence

    Returns:
        pos_embed: [length, embed_dim] positional embedding
    """
    assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos encoding"

    # Position indices
    pos = np.arange(length, dtype=np.float32)

    # Frequency bands
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (embed_dim/2,)

    # Outer product: [length, embed_dim/2]
    out = np.einsum('l,d->ld', pos, omega)

    # Sin and cos
    emb_sin = np.sin(out)  # [length, embed_dim/2]
    emb_cos = np.cos(out)  # [length, embed_dim/2]

    # Concatenate
    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # [length, embed_dim]

    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate positional embedding from explicit positions.

    Args:
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)

    Returns:
        pos_embed: [M, embed_dim]
    """
    assert embed_dim % 2 == 0

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
