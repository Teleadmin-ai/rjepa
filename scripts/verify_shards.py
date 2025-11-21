#!/usr/bin/env python3
"""
Verify that shard format is compatible with LatentDataset.
Quick validation before starting training.
"""

import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rjepa.jepa.dataset import LatentDataset, LatentDatasetMultiShard


def verify_shards(config_path: str = "configs/rjepa/train.yaml"):
    """Verify shard loading and dataset creation."""
    print("=" * 80)
    print("SHARD FORMAT VERIFICATION")
    print("=" * 80)
    print()

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_dir = Path(config['data']['train_latents_dir'])
    print(f"[CONFIG] Train latents dir: {train_dir}")

    # Check directory exists
    if not train_dir.exists():
        print(f"[ERROR] Directory does not exist: {train_dir}")
        return False

    # Count shards
    parquet_files = list(train_dir.glob("shard-*.parquet"))
    safetensors_files = list(train_dir.glob("shard-*.safetensors"))

    print(f"[FILES] Found {len(parquet_files)} parquet files")
    print(f"[FILES] Found {len(safetensors_files)} safetensors files")

    if len(parquet_files) != len(safetensors_files):
        print(f"[ERROR] Mismatch: {len(parquet_files)} parquet vs {len(safetensors_files)} safetensors")
        return False

    if len(parquet_files) == 0:
        print("[ERROR] No shard files found!")
        return False

    print(f"[OK] Found {len(parquet_files)} complete shards")
    print()

    # Try loading dataset (LatentDataset expects directory with both .parquet and .safetensors)
    print(f"[TEST] Loading dataset from directory...")
    try:
        dataset = LatentDataset(
            latents_dir=str(train_dir),
            device="cpu",  # Use CPU for testing
            max_samples=None  # Load all
        )
        print(f"[OK] Dataset loaded: {len(dataset)} total samples")

        # Test accessing first sample
        sample = dataset[0]
        latents, domain_id = sample
        print(f"[OK] First sample shape: {latents.shape}")
        print(f"[OK] Domain ID: {domain_id}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Try creating DataLoader
    print(f"[TEST] Creating DataLoader (batch_size=4)...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 for testing
            drop_last=False
        )

        # Try iterating first batch
        batch = next(iter(dataloader))
        latents_batch, domain_ids_batch = batch
        print(f"[OK] DataLoader working!")
        print(f"[OK] Batch latents shape: {latents_batch.shape}")
        print(f"[OK] Batch domain IDs shape: {domain_ids_batch.shape}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to create DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Shard directory: {train_dir}")
    print(f"Total shards: {len(parquet_files)}")
    print(f"Total samples: {len(dataset)}")
    print(f"LatentDataset: Working")
    print(f"DataLoader: Working")
    print()
    print("[SUCCESS] All checks passed! Ready for training.")
    print("=" * 80)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify shard format compatibility")
    parser.add_argument("--config", default="configs/rjepa/train.yaml", help="Training config path")
    args = parser.parse_args()

    success = verify_shards(args.config)
    sys.exit(0 if success else 1)
