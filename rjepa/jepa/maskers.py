"""
Masking strategies for R-JEPA.

Inspired by V-JEPA multiblock masking, adapted for reasoning steps.
Instead of spatial blocks, we mask temporal/logical blocks of reasoning steps.
"""
import logging
import torch
import numpy as np
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class RandomMasker:
    """
    Random uniform masking of steps.

    Simple baseline: each step has independent probability of being masked.
    """

    def __init__(
        self,
        mask_ratio: float = 0.5,
    ):
        """
        Initialize random masker.

        Args:
            mask_ratio: Probability of masking each step (0.0-1.0)
        """
        self.mask_ratio = mask_ratio
        logger.info(f"RandomMasker initialized (mask_ratio={mask_ratio})")

    def __call__(
        self,
        batch_size: int,
        num_steps: int,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random masks.

        Args:
            batch_size: Batch size
            num_steps: Number of steps per sequence
            device: Device for tensors

        Returns:
            (context_mask, target_mask) where each is [B, S] boolean tensor
            - context_mask: True for steps to use as context
            - target_mask: True for steps to predict
        """
        # Random mask
        mask = torch.rand(batch_size, num_steps, device=device) > self.mask_ratio

        # Context = masked steps (complement)
        context_mask = mask

        # Target = steps to predict (the masked ones)
        target_mask = ~mask

        return context_mask, target_mask


class ContiguousMasker:
    """
    Contiguous block masking.

    Masks a contiguous span of reasoning steps (middle of the sequence).
    This is the RECOMMENDED strategy for R-JEPA as it forces learning
    logical flow and step dependencies.

    Philosophy: Masking the middle forces the model to learn how to
    bridge from premises to conclusions, which is the essence of reasoning.
    """

    def __init__(
        self,
        min_mask_ratio: float = 0.3,
        max_mask_ratio: float = 0.7,
        num_blocks: int = 1,
    ):
        """
        Initialize contiguous masker.

        Args:
            min_mask_ratio: Minimum fraction to mask
            max_mask_ratio: Maximum fraction to mask
            num_blocks: Number of contiguous blocks to mask
        """
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.num_blocks = num_blocks

        logger.info(
            f"ContiguousMasker initialized "
            f"(mask_ratio={min_mask_ratio}-{max_mask_ratio}, "
            f"num_blocks={num_blocks})"
        )

    def __call__(
        self,
        batch_size: int,
        num_steps: int,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate contiguous block masks.

        Args:
            batch_size: Batch size
            num_steps: Number of steps per sequence
            device: Device for tensors

        Returns:
            (context_mask, target_mask)
        """
        context_mask = torch.ones(batch_size, num_steps, dtype=torch.bool, device=device)

        for b in range(batch_size):
            # Sample mask ratio for this sample
            mask_ratio = np.random.uniform(self.min_mask_ratio, self.max_mask_ratio)
            total_mask_steps = int(num_steps * mask_ratio)

            if total_mask_steps == 0:
                continue

            # Divide into num_blocks
            block_size = total_mask_steps // self.num_blocks

            for _ in range(self.num_blocks):
                if block_size == 0:
                    continue

                # Sample start position (avoid edges)
                max_start = max(1, num_steps - block_size - 1)
                start_idx = np.random.randint(0, max_start + 1)
                end_idx = min(start_idx + block_size, num_steps)

                # Mask this block
                context_mask[b, start_idx:end_idx] = False

        target_mask = ~context_mask

        return context_mask, target_mask


class HierarchicalMasker:
    """
    Hierarchical masking with emphasis on middle steps.

    Reasoning steps have different importance:
    - Step 1 (premise): usually visible (high context)
    - Middle steps (reasoning): heavily masked (main prediction target)
    - Final step (conclusion): sometimes visible, sometimes masked

    This reflects the reality that conclusions depend on intermediate reasoning.
    """

    def __init__(
        self,
        intro_keep_prob: float = 0.8,
        middle_mask_prob: float = 0.7,
        conclusion_keep_prob: float = 0.5,
    ):
        """
        Initialize hierarchical masker.

        Args:
            intro_keep_prob: Probability of keeping first step visible
            middle_mask_prob: Probability of masking middle steps
            conclusion_keep_prob: Probability of keeping last step visible
        """
        self.intro_keep_prob = intro_keep_prob
        self.middle_mask_prob = middle_mask_prob
        self.conclusion_keep_prob = conclusion_keep_prob

        logger.info(
            f"HierarchicalMasker initialized "
            f"(intro={intro_keep_prob}, middle_mask={middle_mask_prob}, "
            f"conclusion={conclusion_keep_prob})"
        )

    def __call__(
        self,
        batch_size: int,
        num_steps: int,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate hierarchical masks.

        Args:
            batch_size: Batch size
            num_steps: Number of steps per sequence
            device: Device for tensors

        Returns:
            (context_mask, target_mask)
        """
        context_mask = torch.zeros(batch_size, num_steps, dtype=torch.bool, device=device)

        for b in range(batch_size):
            # First step (intro/premise)
            if np.random.rand() < self.intro_keep_prob:
                context_mask[b, 0] = True

            # Last step (conclusion)
            if num_steps > 1 and np.random.rand() < self.conclusion_keep_prob:
                context_mask[b, -1] = True

            # Middle steps
            if num_steps > 2:
                for s in range(1, num_steps - 1):
                    if np.random.rand() > self.middle_mask_prob:
                        context_mask[b, s] = True

        target_mask = ~context_mask

        return context_mask, target_mask


class MaskCollator:
    """
    Collator that applies masking on-the-fly during batch creation.

    Usage:
        dataset = LatentDataset(...)
        masker = ContiguousMasker(...)
        collator = MaskCollator(masker)

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=collator,
        )
    """

    def __init__(
        self,
        masker,
        device: str = "cpu",
    ):
        """
        Initialize mask collator.

        Args:
            masker: Masker instance (RandomMasker, ContiguousMasker, etc.)
            device: Device for tensors
        """
        self.masker = masker
        self.device = device

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, ...]],
    ) -> dict:
        """
        Collate batch and apply masking.

        Args:
            batch: List of (H, domain_id, ...) tuples from dataset

        Returns:
            Dict with keys:
                - latents: [B, S, D] tensor
                - domain_ids: [B] tensor
                - context_mask: [B, S] boolean
                - target_mask: [B, S] boolean
        """
        # Stack latents
        latents = torch.stack([item[0] for item in batch])  # [B, S, D]

        # Stack domain IDs if present
        domain_ids = None
        if len(batch[0]) > 1:
            domain_ids = torch.tensor([item[1] for item in batch])  # [B]

        batch_size, num_steps, hidden_dim = latents.shape

        # Generate masks
        context_mask, target_mask = self.masker(
            batch_size=batch_size,
            num_steps=num_steps,
            device=self.device,
        )

        return {
            "latents": latents,
            "domain_ids": domain_ids,
            "context_mask": context_mask,
            "target_mask": target_mask,
        }


# Factory function
def create_masker(config: dict):
    """
    Create masker from config dict.

    Args:
        config: Dict with keys:
            - type: "random" | "contiguous" | "hierarchical"
            - ... (type-specific params)

    Returns:
        Masker instance
    """
    masker_type = config.get("type", "contiguous")

    if masker_type == "random":
        return RandomMasker(
            mask_ratio=config.get("mask_ratio", 0.5),
        )
    elif masker_type == "contiguous":
        return ContiguousMasker(
            min_mask_ratio=config.get("min_mask_ratio", 0.3),
            max_mask_ratio=config.get("max_mask_ratio", 0.7),
            num_blocks=config.get("num_blocks", 1),
        )
    elif masker_type == "hierarchical":
        return HierarchicalMasker(
            intro_keep_prob=config.get("intro_keep_prob", 0.8),
            middle_mask_prob=config.get("middle_mask_prob", 0.7),
            conclusion_keep_prob=config.get("conclusion_keep_prob", 0.5),
        )
    else:
        raise ValueError(f"Unknown masker type: {masker_type}")
