"""
Sharding utilities for large datasets.

Splits datasets into manageable shards for:
- Memory efficiency (don't load all data at once)
- Parallel processing
- Incremental updates
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
import math

logger = logging.getLogger(__name__)


class DatasetSharding:
    """Shard large datasets into smaller chunks."""

    @staticmethod
    def shard_records(
        records: List[Dict[str, Any]],
        shard_size: int = 10000,
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Split records into shards.

        Args:
            records: List of records to shard
            shard_size: Number of records per shard

        Yields:
            Shards (sublists of records)
        """
        num_shards = math.ceil(len(records) / shard_size)

        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(records))
            shard = records[start_idx:end_idx]

            logger.debug(
                f"Shard {i}/{num_shards}: {len(shard)} records "
                f"[{start_idx}:{end_idx}]"
            )

            yield shard

    @staticmethod
    def get_shard_path(
        base_dir: Path,
        shard_idx: int,
        prefix: str = "shard",
        extension: str = ".parquet",
    ) -> Path:
        """
        Get path for a shard file.

        Args:
            base_dir: Base directory for shards
            shard_idx: Shard index (0-based)
            prefix: Filename prefix
            extension: File extension

        Returns:
            Path object for shard file
        """
        filename = f"{prefix}-{shard_idx:04d}{extension}"
        return base_dir / filename

    @staticmethod
    def save_sharded_dataset(
        records: List[Dict[str, Any]],
        output_dir: Path,
        shard_size: int = 10000,
        prefix: str = "shard",
        save_fn=None,
    ) -> List[Path]:
        """
        Save dataset as sharded files.

        Args:
            records: List of records to save
            output_dir: Output directory for shards
            shard_size: Records per shard
            prefix: Shard filename prefix
            save_fn: Optional custom save function (records, path) -> None
                If None, uses default parquet writer

        Returns:
            List of shard file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_fn is None:
            from rjepa.utils.io import ParquetIO
            save_fn = ParquetIO.write

        shard_paths = []

        for shard_idx, shard in enumerate(
            DatasetSharding.shard_records(records, shard_size)
        ):
            shard_path = DatasetSharding.get_shard_path(
                output_dir, shard_idx, prefix
            )

            save_fn(shard, shard_path)
            shard_paths.append(shard_path)

        logger.info(
            f"Saved {len(records)} records to {len(shard_paths)} shards "
            f"in {output_dir}"
        )

        return shard_paths

    @staticmethod
    def load_sharded_dataset(
        shard_dir: Path,
        pattern: str = "shard-*.parquet",
        load_fn=None,
    ) -> List[Dict[str, Any]]:
        """
        Load all shards from directory.

        Args:
            shard_dir: Directory containing shards
            pattern: Glob pattern for shard files
            load_fn: Optional custom load function (path) -> List[Dict]
                If None, uses default parquet reader

        Returns:
            List of all records from all shards
        """
        if load_fn is None:
            from rjepa.utils.io import ParquetIO
            load_fn = ParquetIO.read

        shard_paths = sorted(shard_dir.glob(pattern))

        if not shard_paths:
            logger.warning(f"No shards found in {shard_dir} matching {pattern}")
            return []

        all_records = []
        for shard_path in shard_paths:
            records = load_fn(shard_path)
            all_records.extend(records)

        logger.info(
            f"Loaded {len(all_records)} records from {len(shard_paths)} shards "
            f"in {shard_dir}"
        )

        return all_records

    @staticmethod
    def iter_sharded_dataset(
        shard_dir: Path,
        pattern: str = "shard-*.parquet",
        load_fn=None,
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate over shards without loading all into memory.

        Args:
            shard_dir: Directory containing shards
            pattern: Glob pattern for shard files
            load_fn: Optional custom load function

        Yields:
            Records from each shard
        """
        if load_fn is None:
            from rjepa.utils.io import ParquetIO
            load_fn = ParquetIO.read

        shard_paths = sorted(shard_dir.glob(pattern))

        if not shard_paths:
            logger.warning(f"No shards found in {shard_dir} matching {pattern}")
            return

        for shard_idx, shard_path in enumerate(shard_paths):
            logger.debug(f"Loading shard {shard_idx + 1}/{len(shard_paths)}: {shard_path.name}")
            records = load_fn(shard_path)
            yield records


class LatentSharding:
    """Specialized sharding for latent tensors + metadata."""

    @staticmethod
    def save_latent_shard(
        metadata_records: List[Dict[str, Any]],
        latents_dict: Dict[str, Any],  # cot_id -> tensor
        output_dir: Path,
        shard_idx: int,
    ) -> tuple[Path, Path]:
        """
        Save one shard of latents (metadata parquet + tensors safetensors).

        Args:
            metadata_records: List of metadata dicts
            latents_dict: Dict mapping cot_id to latent tensors
            output_dir: Output directory
            shard_idx: Shard index

        Returns:
            (metadata_path, latents_path)
        """
        from rjepa.utils.io import ParquetIO, SafeTensorsIO

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path = DatasetSharding.get_shard_path(
            output_dir, shard_idx, prefix="shard", extension=".parquet"
        )
        ParquetIO.write(metadata_records, metadata_path)

        # Save latents
        latents_path = DatasetSharding.get_shard_path(
            output_dir, shard_idx, prefix="shard", extension=".safetensors"
        )
        SafeTensorsIO.save_latents(latents_dict, latents_path)

        logger.info(
            f"Saved shard {shard_idx}: {len(metadata_records)} metadata records, "
            f"{len(latents_dict)} latent tensors"
        )

        return metadata_path, latents_path

    @staticmethod
    def load_latent_shard(
        shard_dir: Path,
        shard_idx: int,
        device: str = "cpu",
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load one shard of latents.

        Args:
            shard_dir: Directory containing shards
            shard_idx: Shard index to load
            device: Device for tensors

        Returns:
            (metadata_records, latents_dict)
        """
        from rjepa.utils.io import ParquetIO, SafeTensorsIO

        # Load metadata
        metadata_path = DatasetSharding.get_shard_path(
            shard_dir, shard_idx, prefix="shard", extension=".parquet"
        )
        metadata_records = ParquetIO.read(metadata_path)

        # Load latents
        latents_path = DatasetSharding.get_shard_path(
            shard_dir, shard_idx, prefix="shard", extension=".safetensors"
        )
        latents_dict = SafeTensorsIO.load_latents(latents_path, device=device)

        logger.info(
            f"Loaded shard {shard_idx}: {len(metadata_records)} metadata records, "
            f"{len(latents_dict)} latent tensors"
        )

        return metadata_records, latents_dict

    @staticmethod
    def iter_latent_shards(
        shard_dir: Path,
        device: str = "cpu",
    ) -> Iterator[tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Iterate over all latent shards.

        Args:
            shard_dir: Directory containing shards
            device: Device for tensors

        Yields:
            (metadata_records, latents_dict) for each shard
        """
        # Find all metadata shards
        metadata_shards = sorted(shard_dir.glob("shard-*.parquet"))

        if not metadata_shards:
            logger.warning(f"No latent shards found in {shard_dir}")
            return

        for shard_idx, metadata_path in enumerate(metadata_shards):
            logger.debug(f"Loading latent shard {shard_idx + 1}/{len(metadata_shards)}")
            metadata_records, latents_dict = LatentSharding.load_latent_shard(
                shard_dir, shard_idx, device=device
            )
            yield metadata_records, latents_dict
