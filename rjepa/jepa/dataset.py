"""
Dataset for loading latent sequences for R-JEPA training.

Loads pre-extracted latents from parquet (metadata) + safetensors (tensors).
"""
import logging
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List

from rjepa.data.sharding import LatentSharding
from rjepa.utils.io import ParquetIO, SafeTensorsIO

logger = logging.getLogger(__name__)


# Domain mapping (domain string -> ID)
DOMAIN_MAP = {
    "math": 0,
    "code": 1,
    "logic": 2,
    "mixed": 3,
    "unknown": 4,
}


class LatentDataset(Dataset):
    """
    Dataset for latent sequences.

    Loads from sharded parquet + safetensors files.
    Each sample is a (H, domain_id) tuple where:
    - H: [num_steps, hidden_size] latent tensor
    - domain_id: int (domain class)
    """

    def __init__(
        self,
        latents_dir: Path,
        device: str = "cpu",
        max_samples: Optional[int] = None,
        domain_map: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize dataset.

        Args:
            latents_dir: Directory containing latent shards
                Should have files like:
                    shard-0000.parquet (metadata)
                    shard-0000.safetensors (tensors)
            device: Device to load tensors to
            max_samples: Optional limit on number of samples
            domain_map: Optional custom domain mapping
        """
        self.latents_dir = Path(latents_dir)
        self.device = device
        self.domain_map = domain_map or DOMAIN_MAP

        if not self.latents_dir.exists():
            raise ValueError(f"Latents directory not found: {latents_dir}")

        # Load all metadata and latents
        logger.info(f"Loading latent dataset from {latents_dir}...")
        self.metadata = []
        self.latents = {}

        for metadata_records, latents_dict in LatentSharding.iter_latent_shards(
            self.latents_dir, device=device
        ):
            self.metadata.extend(metadata_records)
            self.latents.update(latents_dict)

        # Apply max_samples limit
        if max_samples is not None and max_samples < len(self.metadata):
            self.metadata = self.metadata[:max_samples]
            # Filter latents
            valid_cot_ids = {m["cot_id"] for m in self.metadata}
            self.latents = {k: v for k, v in self.latents.items() if k in valid_cot_ids}

        logger.info(
            f"Loaded {len(self.metadata)} samples with {len(self.latents)} latent tensors"
        )

        # Sanity check
        if len(self.metadata) != len(self.latents):
            logger.warning(
                f"Mismatch: {len(self.metadata)} metadata records vs "
                f"{len(self.latents)} latent tensors"
            )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get one sample.

        Args:
            idx: Sample index

        Returns:
            (H, domain_id) tuple where:
                - H: [num_steps, hidden_size] tensor
                - domain_id: int
        """
        metadata = self.metadata[idx]
        cot_id = metadata["cot_id"]

        # Get latent tensor
        H = self.latents[cot_id]  # [num_steps, hidden_size]

        # Get domain ID
        domain = metadata.get("domain", "unknown")
        domain_id = self.domain_map.get(domain, self.domain_map.get("unknown", 4))

        return H, domain_id

    def get_stats(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dict with statistics
        """
        # Count by domain
        domain_counts = {}
        for m in self.metadata:
            domain = m.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Num steps distribution
        num_steps_list = [m["num_steps"] for m in self.metadata]

        stats = {
            "num_samples": len(self),
            "domain_counts": domain_counts,
            "num_steps": {
                "min": min(num_steps_list) if num_steps_list else 0,
                "max": max(num_steps_list) if num_steps_list else 0,
                "mean": sum(num_steps_list) / len(num_steps_list) if num_steps_list else 0,
            },
            "hidden_size": self.metadata[0]["hidden_size"] if self.metadata else 0,
        }

        return stats


class LatentDatasetMultiShard(Dataset):
    """
    Memory-efficient dataset that loads shards on-the-fly.

    Instead of loading all shards at once, this dataset loads them
    lazily as needed. Useful for very large datasets.
    """

    def __init__(
        self,
        latents_dir: Path,
        device: str = "cpu",
        domain_map: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize multi-shard dataset.

        Args:
            latents_dir: Directory containing latent shards
            device: Device to load tensors to
            domain_map: Optional custom domain mapping
        """
        self.latents_dir = Path(latents_dir)
        self.device = device
        self.domain_map = domain_map or DOMAIN_MAP

        if not self.latents_dir.exists():
            raise ValueError(f"Latents directory not found: {latents_dir}")

        # Find all shard files
        self.shard_paths = sorted(self.latents_dir.glob("shard-*.parquet"))

        if not self.shard_paths:
            raise ValueError(f"No shards found in {latents_dir}")

        # Load metadata for all shards (lightweight)
        logger.info(f"Loading metadata from {len(self.shard_paths)} shards...")
        self.metadata = []
        self.shard_offsets = [0]  # Cumulative offsets

        for shard_path in self.shard_paths:
            metadata_records = ParquetIO.read(shard_path)
            self.metadata.extend(metadata_records)
            self.shard_offsets.append(len(self.metadata))

        logger.info(f"Loaded metadata for {len(self.metadata)} samples")

        # Cache for current shard
        self.current_shard_idx = None
        self.current_latents = None

    def _load_shard(self, shard_idx: int):
        """Load a specific shard into cache."""
        if shard_idx == self.current_shard_idx:
            return  # Already loaded

        logger.debug(f"Loading shard {shard_idx}...")
        _, latents = LatentSharding.load_latent_shard(
            self.latents_dir, shard_idx, device=self.device
        )

        self.current_shard_idx = shard_idx
        self.current_latents = latents

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get one sample (loads shard if needed).

        Args:
            idx: Sample index

        Returns:
            (H, domain_id) tuple
        """
        # Find which shard this sample belongs to
        shard_idx = 0
        for i, offset in enumerate(self.shard_offsets[1:], start=0):
            if idx < offset:
                shard_idx = i
                break

        # Load shard if not already loaded
        self._load_shard(shard_idx)

        # Get sample
        metadata = self.metadata[idx]
        cot_id = metadata["cot_id"]

        H = self.current_latents[cot_id]

        domain = metadata.get("domain", "unknown")
        domain_id = self.domain_map.get(domain, self.domain_map.get("unknown", 4))

        return H, domain_id
