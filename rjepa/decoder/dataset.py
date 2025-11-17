"""
Latent Decoder Dataset.

Loads (latent, text) pairs for training the decoder to generate text from R-JEPA latents.

Philosophy:
- Uses pre-computed latents from frozen R-JEPA (world model as ground truth)
- Only uses validated CoTs (is_valid=True)
- Tokenizes step text for language modeling
"""
import logging
from pathlib import Path
from typing import Optional, Dict, List
import torch
from torch.utils.data import Dataset
import pandas as pd
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class LatentTextDataset(Dataset):
    """
    Dataset for training LatentDecoder.

    Loads (latent, text) pairs where:
    - latent: [latent_dim] vector from R-JEPA layer -2
    - text: step text from validated CoT

    Philosophy: R-JEPA is frozen, decoder learns to verbalize latents.
    """

    def __init__(
        self,
        latents_dir: Path,
        cots_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 256,
        llm_tag: str = "qwen3-8b",
        split: str = "train",
    ):
        """
        Initialize dataset.

        Args:
            latents_dir: Directory containing latent shards (parquet + safetensors)
            cots_path: Path to CoT parquet file (with text steps)
            tokenizer: HuggingFace tokenizer
            max_seq_len: Maximum sequence length for tokenization
            llm_tag: LLM tag (e.g., "qwen3-8b")
            split: "train" or "val"
        """
        self.latents_dir = Path(latents_dir)
        self.cots_path = Path(cots_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.llm_tag = llm_tag
        self.split = split

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load CoT metadata (text steps)
        logger.info(f"Loading CoT metadata from {self.cots_path}")
        self.cots_df = pd.read_parquet(self.cots_path)

        # Filter by is_valid=True (only validated reasoning)
        self.cots_df = self.cots_df[self.cots_df["is_valid"] == True]
        logger.info(f"Filtered to {len(self.cots_df)} validated CoTs")

        # Load latent metadata
        latent_path = self.latents_dir / llm_tag / split
        if not latent_path.exists():
            raise FileNotFoundError(
                f"Latent path not found: {latent_path}. "
                f"Run 'python -m rjepa.pipeline.build_latents --llm {llm_tag} --split {split}' first."
            )

        # Find all latent shards
        self.latent_shards = sorted(latent_path.glob("shard-*.parquet"))
        if not self.latent_shards:
            raise FileNotFoundError(f"No latent shards found in {latent_path}")

        logger.info(f"Found {len(self.latent_shards)} latent shards")

        # Load all latent metadata (lazy load actual tensors)
        self.latents_meta = []
        for shard_path in self.latent_shards:
            df = pd.read_parquet(shard_path)
            self.latents_meta.append(df)

        self.latents_meta = pd.concat(self.latents_meta, ignore_index=True)
        logger.info(f"Loaded {len(self.latents_meta)} latent records")

        # Join CoTs with latents on (problem_id, cot_id)
        # This gives us (latent_metadata, step_text) pairs
        merged = self.cots_df.merge(
            self.latents_meta,
            on=["problem_id", "cot_id"],
            how="inner",
            suffixes=("_cot", "_lat"),
        )

        # Explode steps (one row per step)
        # CoT has steps: List[str], latent has num_steps: int
        # We need to match each step text with its corresponding latent
        self.samples = []
        for _, row in merged.iterrows():
            steps = row["steps"]  # List[str]
            num_steps = row["num_steps"]

            # Ensure consistency
            if len(steps) != num_steps:
                logger.warning(
                    f"Mismatch: cot_id={row['cot_id']} has {len(steps)} steps "
                    f"but latent has {num_steps} steps. Skipping."
                )
                continue

            # Create one sample per step
            for step_idx, step_text in enumerate(steps):
                self.samples.append(
                    {
                        "cot_id": row["cot_id"],
                        "problem_id": row["problem_id"],
                        "step_idx": step_idx,
                        "step_text": step_text,
                        "latent_shard": row["shard_idx"]
                        if "shard_idx" in row
                        else 0,  # Which shard file
                        "latent_idx": row["latent_idx"]
                        if "latent_idx" in row
                        else _,  # Index within shard
                    }
                )

        logger.info(f"Created {len(self.samples)} (latent, text) samples")

        # Cache for loaded latent tensors (lazy loading)
        self._latent_cache = {}

    def _load_latent(self, shard_idx: int, latent_idx: int) -> torch.Tensor:
        """
        Load a single latent vector from safetensors file.

        Args:
            shard_idx: Shard file index
            latent_idx: Index within shard

        Returns:
            [latent_dim] tensor
        """
        cache_key = (shard_idx, latent_idx)
        if cache_key in self._latent_cache:
            return self._latent_cache[cache_key]

        # Load safetensors file
        shard_path = self.latent_shards[shard_idx]
        safetensors_path = shard_path.with_suffix(".safetensors")

        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"Safetensors file not found: {safetensors_path}. "
                f"Ensure latents were built with safetensors output."
            )

        # Load tensor (using safetensors library)
        try:
            from safetensors import safe_open

            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                # Assuming key format: "latent_{idx}"
                latent = f.get_tensor(f"latent_{latent_idx}")

            # Cache it
            self._latent_cache[cache_key] = latent
            return latent

        except Exception as e:
            logger.error(f"Failed to load latent from {safetensors_path}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single (latent, text) sample.

        Returns:
            {
              "latent": [latent_dim] tensor,
              "input_ids": [seq_len] tensor,
              "labels": [seq_len] tensor (shifted input_ids with -100 for pad)
            }
        """
        sample = self.samples[idx]

        # Load latent
        latent = self._load_latent(
            sample["latent_shard"], sample["latent_idx"]
        )  # [latent_dim]

        # Tokenize step text
        step_text = sample["step_text"]
        encoding = self.tokenizer(
            step_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)  # [seq_len]

        # Labels for language modeling (shifted input_ids)
        # Replace pad tokens with -100 (ignored by cross_entropy)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "latent": latent,
            "input_ids": input_ids,
            "labels": labels,
        }


def create_decoder_dataloaders(
    latents_dir: Path,
    cots_train_path: Path,
    cots_val_path: Optional[Path],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_seq_len: int = 256,
    llm_tag: str = "qwen3-8b",
    num_workers: int = 4,
) -> tuple:
    """
    Create train and validation dataloaders for LatentDecoder.

    Args:
        latents_dir: Directory containing latent shards
        cots_train_path: Path to train CoT parquet
        cots_val_path: Path to val CoT parquet (optional)
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_seq_len: Max sequence length
        llm_tag: LLM tag
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader) tuple
    """
    from torch.utils.data import DataLoader

    # Train dataset
    train_dataset = LatentTextDataset(
        latents_dir=latents_dir,
        cots_path=cots_train_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        llm_tag=llm_tag,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Val dataset (optional)
    val_loader = None
    if cots_val_path is not None:
        val_dataset = LatentTextDataset(
            latents_dir=latents_dir,
            cots_path=cots_val_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            llm_tag=llm_tag,
            split="val",
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    logger.info(
        f"Created dataloaders: train={len(train_dataset)} samples, "
        f"val={len(val_dataset) if val_loader else 0} samples"
    )

    return train_loader, val_loader
