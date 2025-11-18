# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# ADAPTED FOR R-JEPA: multiblock3d → multiblock1d
# Adapts V-JEPA's spatial/temporal block masking to 1D reasoning step sequences

import math
from multiprocessing import Value
from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):
    """
    Mask Collator for R-JEPA.

    Generates context and target masks for reasoning step sequences.
    Adapted from V-JEPA's multiblock3d masking (spatial+temporal) to 1D (sequential).

    Instead of masking 3D blocks (time × height × width),
    we mask 1D blocks (contiguous sequences of reasoning steps).
    """

    def __init__(
        self,
        cfgs_mask,           # List of mask config dicts
        max_seq_len=512,     # Maximum sequence length
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _MaskGenerator(
                max_seq_len=max_seq_len,
                pred_mask_scale=m.get('pred_mask_scale', (0.2, 0.8)),
                aspect_ratio=m.get('aspect_ratio', (1.0, 1.0)),  # Not used in 1D
                num_blocks=m.get('num_blocks', 1),
                max_context_len_ratio=m.get('max_context_len_ratio', 1.0),
                max_keep=m.get('max_keep', None),
            )
            self.mask_generators.append(mask_generator)

    def step(self):
        """Increment iteration counter (for deterministic sampling across workers)"""
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):
        """
        Collate batch and generate masks.

        Args:
            batch: List of [S, D] latent tensors (one per problem)

        Returns:
            collated_batch: [B, S, D] stacked latents
            collated_masks_enc: List of [B, K_context] context mask indices
            collated_masks_pred: List of [B, K_target] target mask indices
        """
        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_pred, collated_masks_enc = [], []
        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc, masks_pred = mask_generator(batch_size)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred


class _MaskGenerator(object):
    """
    Single mask generator for 1D reasoning sequences.

    Generates context/target masks by:
    1. Sampling block size (length of contiguous steps to mask)
    2. Sampling block locations (where to place masks)
    3. Creating binary mask (1 = context, 0 = target)
    4. Extracting indices for context and target tokens
    """

    def __init__(
        self,
        max_seq_len=512,
        pred_mask_scale=(0.2, 0.8),    # (min, max) fraction of sequence to predict
        aspect_ratio=(1.0, 1.0),       # Not used in 1D (kept for compatibility)
        num_blocks=1,                  # Number of masked blocks
        max_context_len_ratio=1.0,     # Max fraction of sequence used as context
        max_keep=None,                 # Max number of context tokens to keep
    ):
        super(_MaskGenerator, self).__init__()

        self.max_seq_len = max_seq_len
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio  # Unused in 1D, kept for API compat
        self.num_blocks = num_blocks
        self.max_context_len = max(1, int(max_seq_len * max_context_len_ratio))
        self.max_keep = max_keep

        self._itr_counter = Value('i', -1)  # Shared across worker processes

    def step(self):
        """Increment iteration counter"""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, pred_scale):
        """
        Sample length of contiguous block to mask.

        Args:
            generator: torch.Generator for reproducibility
            pred_scale: (min, max) fraction of sequence to mask

        Returns:
            length: int, number of steps in masked block
        """
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = pred_scale
        mask_scale = min_s + _rand * (max_s - min_s)

        length = max(1, int(self.max_seq_len * mask_scale))
        return length

    def _sample_block_mask(self, block_size):
        """
        Create binary mask with one masked block.

        Args:
            block_size: int, length of block to mask

        Returns:
            mask: [max_seq_len] binary tensor (1 = context, 0 = target)
        """
        # Sample random starting position for masked block
        start = torch.randint(0, self.max_seq_len - block_size + 1, (1,)).item()

        mask = torch.ones(self.max_seq_len, dtype=torch.int32)
        mask[start:start+block_size] = 0  # Mark target steps as 0

        # Context mask only spans first X steps (if limited)
        if self.max_context_len < self.max_seq_len:
            mask[self.max_context_len:] = 0

        return mask

    def __call__(self, batch_size):
        """
        Generate encoder and predictor masks for a batch.

        Process:
        1. Sample block size using deterministic seed (same across batch)
        2. Sample block locations per sample (different per sample)
        3. Return context and target masks

        Args:
            batch_size: int

        Returns:
            collated_masks_enc: [B, K_context] indices of context steps
            collated_masks_pred: [B, K_target] indices of target steps
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        # Sample block size (shared across batch for this iteration)
        block_size = self._sample_block_size(
            generator=g,
            pred_scale=self.pred_mask_scale
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.max_seq_len

        for _ in range(batch_size):
            # Ensure we don't create empty context
            empty_context = True
            while empty_context:
                # Start with full context
                mask_e = torch.ones(self.max_seq_len, dtype=torch.int32)

                # Apply num_blocks masked regions
                for _ in range(self.num_blocks):
                    mask_e *= self._sample_block_mask(block_size)

                # Extract indices
                mask_p = torch.argwhere(mask_e == 0).squeeze()  # Target (predict)
                mask_e = torch.nonzero(mask_e).squeeze()        # Context (encode)

                # Check if context is non-empty
                empty_context = (len(mask_e) == 0) if mask_e.dim() > 0 else (mask_e.numel() == 0)

                if not empty_context:
                    # Track minimum sizes for batching
                    if mask_p.dim() == 0:
                        mask_p = mask_p.unsqueeze(0)
                    if mask_e.dim() == 0:
                        mask_e = mask_e.unsqueeze(0)

                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))

                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        # Apply max_keep limit if specified
        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        # Truncate all masks to minimum size (for batching)
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred
