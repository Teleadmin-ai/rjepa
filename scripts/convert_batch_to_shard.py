"""
Convert batch_*.pkl.gz files to shard_*.parquet + shard_*.safetensors format.

This script converts the output from extract_latents_optimized.py (batch_*.pkl.gz)
to the format expected by LatentDataset (shard_*.parquet + shard_*.safetensors).
"""
import pickle
import gzip
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from safetensors.torch import save_file
import argparse


def convert_batch_to_shard(input_dir: Path, output_dir: Path, shard_size: int = 1000):
    """
    Convert batch_*.pkl.gz files to shard format.

    Args:
        input_dir: Directory containing batch_*.pkl.gz files
        output_dir: Directory to save shard_*.parquet and shard_*.safetensors
        shard_size: Number of samples per shard
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all batch files
    batch_files = sorted(input_dir.glob("batch_*.pkl.gz"))

    if not batch_files:
        raise ValueError(f"No batch_*.pkl.gz files found in {input_dir}")

    print(f"Found {len(batch_files)} batch files")

    # Accumulators for current shard
    current_shard_metadata = []
    current_shard_latents = {}
    shard_idx = 0
    total_samples = 0

    for batch_file in tqdm(batch_files, desc="Converting batches"):
        # Load batch
        with gzip.open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)

        # batch_data is a list of dicts with actual keys:
        # 'status', 'problem_id', 'domain', 'subdomain', 'num_steps', 'hidden_size',
        # 'latent_shape', 'cot_text', 'steps', 'step_boundaries',
        # 'latents_compressed', 'latents_size_mb'
        for sample in batch_data:
            # Skip failed extractions
            if sample.get('status') != 'success':
                continue

            # Generate cot_id from problem_id (since cot_id doesn't exist in batch format)
            problem_id = sample['problem_id']
            cot_id = problem_id  # Each problem has one CoT, so cot_id = problem_id

            # Decompress latents
            latents_compressed = sample['latents_compressed']
            latents_decompressed = gzip.decompress(latents_compressed)

            # Convert to numpy array (stored as float16 for efficiency)
            latents_array = np.frombuffer(latents_decompressed, dtype=np.float16)

            # Calculate actual shape from decompressed data
            hidden_size = sample['hidden_size']  # 4096
            num_elements = len(latents_array)
            actual_num_steps = num_elements // hidden_size

            # Reshape and convert to float32 for training
            latents_tensor = torch.from_numpy(latents_array.copy()).reshape(actual_num_steps, hidden_size).to(torch.float32)

            # Metadata record (use actual calculated num_steps, not potentially incorrect metadata)
            metadata_record = {
                'cot_id': cot_id,
                'problem_id': problem_id,
                'domain': sample.get('domain', 'unknown'),
                'subdomain': sample.get('subdomain', ''),
                'num_steps': actual_num_steps,  # Use calculated value
                'hidden_size': hidden_size,
            }
            current_shard_metadata.append(metadata_record)

            # Latent tensor
            current_shard_latents[cot_id] = latents_tensor

            total_samples += 1

            # Save shard if it reached the size limit
            if len(current_shard_metadata) >= shard_size:
                save_shard(
                    output_dir,
                    shard_idx,
                    current_shard_metadata,
                    current_shard_latents
                )

                # Reset accumulators
                current_shard_metadata = []
                current_shard_latents = {}
                shard_idx += 1

    # Save remaining samples
    if current_shard_metadata:
        save_shard(
            output_dir,
            shard_idx,
            current_shard_metadata,
            current_shard_latents
        )
        shard_idx += 1

    print(f"\n✅ Conversion complete!")
    print(f"   Total samples: {total_samples}")
    print(f"   Total shards: {shard_idx}")
    print(f"   Output directory: {output_dir}")

    # Verify
    verify_shards(output_dir, total_samples)


def save_shard(output_dir: Path, shard_idx: int, metadata: list, latents: dict):
    """Save one shard to parquet + safetensors."""
    shard_name = f"shard-{shard_idx:04d}"

    # Save metadata to parquet
    metadata_path = output_dir / f"{shard_name}.parquet"
    df = pd.DataFrame(metadata)
    df.to_parquet(metadata_path, compression='snappy', index=False)

    # Save latents to safetensors
    latents_path = output_dir / f"{shard_name}.safetensors"
    save_file(latents, latents_path)

    print(f"  Saved {shard_name}: {len(metadata)} samples")


def verify_shards(output_dir: Path, expected_total: int):
    """Verify the conversion was successful."""
    parquet_files = sorted(output_dir.glob("shard-*.parquet"))
    safetensors_files = sorted(output_dir.glob("shard-*.safetensors"))

    print(f"\n🔍 Verification:")
    print(f"   Parquet files: {len(parquet_files)}")
    print(f"   SafeTensors files: {len(safetensors_files)}")

    if len(parquet_files) != len(safetensors_files):
        print(f"   ⚠️  Mismatch! {len(parquet_files)} parquet vs {len(safetensors_files)} safetensors")
        return False

    # Count total samples
    total_samples = 0
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        total_samples += len(df)

    print(f"   Total samples in shards: {total_samples}")

    if total_samples == expected_total:
        print(f"   ✅ All samples accounted for!")
        return True
    else:
        print(f"   ⚠️  Expected {expected_total} but found {total_samples}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert batch_*.pkl.gz to shard_*.parquet format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/latents/qwen3-8b/academic"),
        help="Directory containing batch_*.pkl.gz files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/latents/qwen3-8b/academic_shards"),
        help="Directory to save shard files"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of samples per shard"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("BATCH TO SHARD CONVERSION")
    print("=" * 80)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Shard size: {args.shard_size} samples")
    print("=" * 80)

    convert_batch_to_shard(args.input_dir, args.output_dir, args.shard_size)


if __name__ == "__main__":
    main()
