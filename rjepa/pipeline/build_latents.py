"""
Build Latents Pipeline.

Extracts latent representations from CoT text using student LLM.

Input: CoTs (text) from parquet files
Output: Latents (tensors) saved as sharded parquet + safetensors

This is the KEY pipeline for converting text reasoning into latent space.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from dataclasses import asdict
from tqdm import tqdm

from rjepa.config.settings import Settings
from rjepa.data.schemas import ChainOfThought, LatentSequence
from rjepa.llm.adapter import LLMAdapter
from rjepa.utils.io import ParquetIO, save_metadata_json
from rjepa.data.sharding import DatasetSharding, LatentSharding

logger = logging.getLogger(__name__)


def build_latents_from_cots(
    cots_path: Path,
    output_dir: Path,
    llm_config: Dict[str, Any],
    layer_idx: int = -2,
    shard_size: int = 1000,
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Build latents from CoT text file.

    Args:
        cots_path: Path to CoTs parquet file
        output_dir: Output directory for latent shards
        llm_config: LLM configuration dict
            {
              "model_name": "Qwen/Qwen3-8B-Instruct",
              "quantization": "awq-4bit",
              "dtype": "bfloat16",
            }
        layer_idx: Layer to extract latents from (default: -2)
        shard_size: Number of samples per shard
        max_samples: Optional limit on number of CoTs to process
        device: "cuda" or "cpu"

    Returns:
        Statistics dict
    """
    logger.info("=" * 80)
    logger.info("BUILD LATENTS PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input: {cots_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"LLM: {llm_config.get('model_name')}")
    logger.info(f"Layer: {layer_idx}")
    logger.info(f"Shard size: {shard_size}")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CoTs
    logger.info("Loading CoTs from parquet...")
    cots_data = ParquetIO.read(cots_path)

    if max_samples:
        cots_data = cots_data[:max_samples]

    logger.info(f"Loaded {len(cots_data)} CoTs")

    # Initialize LLM adapter
    logger.info("Initializing LLM adapter...")
    llm = LLMAdapter(
        model_name=llm_config.get("model_name", "Qwen/Qwen3-8B-Instruct"),
        device=device,
        dtype=llm_config.get("dtype", "bfloat16"),
        quantization=llm_config.get("quantization", "awq-4bit"),
        layer_to_extract=layer_idx,
    )

    llm_tag = f"{llm.model_name.split('/')[-1].lower()}-{llm_config.get('quantization', 'fp16')}"
    logger.info(f"LLM tag: {llm_tag}")

    # Extract latents for all CoTs
    logger.info("Extracting latents...")
    metadata_records = []
    latents_dict = {}
    failed_count = 0

    for cot_data in tqdm(cots_data, desc="Extracting latents"):
        try:
            # Reconstruct CoT text
            cot = ChainOfThought(**cot_data)
            full_text = "\n".join(cot.steps)

            # Tokenize
            tokens = llm.tokenizer.encode(full_text, return_tensors="pt").to(device)

            # Segment into steps
            from rjepa.llm.step_segmentation import StepSegmentation

            step_boundaries = StepSegmentation.segment_explicit_markers(
                full_text, tokens[0], llm.tokenizer
            )

            if not step_boundaries:
                logger.warning(f"No steps found for CoT {cot.cot_id}, skipping")
                failed_count += 1
                continue

            # Extract latents
            H = llm.extract_latents(tokens, step_boundaries, layer_idx=layer_idx)

            # Create metadata record
            metadata = LatentSequence(
                problem_id=cot.problem_id,
                cot_id=cot.cot_id,
                llm_tag=llm_tag,
                layer_idx=layer_idx,
                hidden_size=H.shape[1],
                num_steps=H.shape[0],
                step_boundaries=step_boundaries,
                domain=cot_data.get("domain", "unknown"),
                subdomain=cot_data.get("subdomain", ""),
            )
            metadata_records.append(asdict(metadata))

            # Store latent tensor (CPU to save GPU memory)
            latents_dict[cot.cot_id] = H.cpu()

        except Exception as e:
            logger.error(f"Failed to extract latents for CoT {cot_data.get('cot_id')}: {e}")
            failed_count += 1
            continue

    logger.info(f"Successfully extracted latents for {len(metadata_records)} CoTs")
    logger.info(f"Failed: {failed_count}")

    # Save as shards
    logger.info("Saving latent shards...")
    shard_idx = 0
    shard_metadata = []
    shard_latents = {}

    for idx, (metadata, (cot_id, latent)) in enumerate(
        zip(metadata_records, latents_dict.items())
    ):
        shard_metadata.append(metadata)
        shard_latents[cot_id] = latent

        # Save shard when full
        if len(shard_metadata) >= shard_size or idx == len(metadata_records) - 1:
            LatentSharding.save_latent_shard(
                shard_metadata, shard_latents, output_dir, shard_idx
            )
            shard_idx += 1
            shard_metadata = []
            shard_latents = {}

    # Save summary metadata
    summary = {
        "llm_tag": llm_tag,
        "layer_idx": layer_idx,
        "hidden_size": llm.hidden_size,
        "num_samples": len(metadata_records),
        "num_shards": shard_idx,
        "shard_size": shard_size,
        "source_cots_path": str(cots_path),
        "failed_count": failed_count,
    }

    summary_path = output_dir / "metadata.json"
    save_metadata_json(summary, summary_path)

    logger.info("=" * 80)
    logger.info("BUILD LATENTS COMPLETE")
    logger.info(f"Total samples: {len(metadata_records)}")
    logger.info(f"Total shards: {shard_idx}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# PREFECT FLOW
# ═════════════════════════════════════════════════════════════════════════════

try:
    from prefect import flow, task

    @task(name="load_cots_task")
    def load_cots_task(cots_path: Path, max_samples: Optional[int] = None):
        """Load CoTs from parquet."""
        cots_data = ParquetIO.read(cots_path)
        if max_samples:
            cots_data = cots_data[:max_samples]
        return cots_data

    @task(name="initialize_llm_task")
    def initialize_llm_task(llm_config: Dict[str, Any], layer_idx: int, device: str):
        """Initialize LLM adapter."""
        llm = LLMAdapter(
            model_name=llm_config.get("model_name", "Qwen/Qwen3-8B-Instruct"),
            device=device,
            dtype=llm_config.get("dtype", "bfloat16"),
            quantization=llm_config.get("quantization", "awq-4bit"),
            layer_to_extract=layer_idx,
        )
        llm_tag = f"{llm.model_name.split('/')[-1].lower()}-{llm_config.get('quantization', 'fp16')}"
        return llm, llm_tag

    @task(name="extract_latents_batch_task")
    def extract_latents_batch_task(
        cots_batch: List[Dict],
        llm: LLMAdapter,
        llm_tag: str,
        layer_idx: int,
        device: str,
    ):
        """Extract latents for a batch of CoTs."""
        from rjepa.llm.step_segmentation import StepSegmentation

        metadata_records = []
        latents_dict = {}

        for cot_data in cots_batch:
            try:
                cot = ChainOfThought(**cot_data)
                full_text = "\n".join(cot.steps)
                tokens = llm.tokenizer.encode(full_text, return_tensors="pt").to(device)

                step_boundaries = StepSegmentation.segment_explicit_markers(
                    full_text, tokens[0], llm.tokenizer
                )

                if not step_boundaries:
                    continue

                H = llm.extract_latents(tokens, step_boundaries, layer_idx=layer_idx)

                metadata = LatentSequence(
                    problem_id=cot.problem_id,
                    cot_id=cot.cot_id,
                    llm_tag=llm_tag,
                    layer_idx=layer_idx,
                    hidden_size=H.shape[1],
                    num_steps=H.shape[0],
                    step_boundaries=step_boundaries,
                    domain=cot_data.get("domain", "unknown"),
                    subdomain=cot_data.get("subdomain", ""),
                )
                metadata_records.append(asdict(metadata))
                latents_dict[cot.cot_id] = H.cpu()

            except Exception as e:
                logger.error(f"Failed: {e}")
                continue

        return metadata_records, latents_dict

    @task(name="save_latents_shards_task")
    def save_latents_shards_task(
        metadata_records: List[Dict],
        latents_dict: Dict[str, torch.Tensor],
        output_dir: Path,
        shard_size: int,
    ):
        """Save latents as shards."""
        shard_idx = 0
        shard_metadata = []
        shard_latents = {}

        for idx, (metadata, (cot_id, latent)) in enumerate(
            zip(metadata_records, latents_dict.items())
        ):
            shard_metadata.append(metadata)
            shard_latents[cot_id] = latent

            if len(shard_metadata) >= shard_size or idx == len(metadata_records) - 1:
                LatentSharding.save_latent_shard(
                    shard_metadata, shard_latents, output_dir, shard_idx
                )
                shard_idx += 1
                shard_metadata = []
                shard_latents = {}

        return shard_idx

    @flow(name="build_latents_flow")
    def build_latents_flow(
        cots_path: str,
        output_dir: str,
        llm_config: Dict[str, Any],
        layer_idx: int = -2,
        shard_size: int = 1000,
        max_samples: Optional[int] = None,
        device: str = "cuda",
    ):
        """
        Prefect flow for building latents.

        Args:
            cots_path: Path to CoTs parquet
            output_dir: Output directory
            llm_config: LLM configuration
            layer_idx: Layer to extract
            shard_size: Samples per shard
            max_samples: Optional limit
            device: "cuda" or "cpu"
        """
        cots_path = Path(cots_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load CoTs
        cots_data = load_cots_task(cots_path, max_samples)

        # Initialize LLM
        llm, llm_tag = initialize_llm_task(llm_config, layer_idx, device)

        # Extract latents (batch processing)
        metadata_records, latents_dict = extract_latents_batch_task(
            cots_data, llm, llm_tag, layer_idx, device
        )

        # Save shards
        num_shards = save_latents_shards_task(
            metadata_records, latents_dict, output_dir, shard_size
        )

        # Save summary
        summary = {
            "llm_tag": llm_tag,
            "layer_idx": layer_idx,
            "hidden_size": llm.hidden_size,
            "num_samples": len(metadata_records),
            "num_shards": num_shards,
            "shard_size": shard_size,
            "source_cots_path": str(cots_path),
        }
        save_metadata_json(summary, output_dir / "metadata.json")

        return summary

except ImportError:
    logger.warning("Prefect not installed, flows not available")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build latents from CoT text")
    parser.add_argument("--cots", type=Path, required=True, help="CoTs parquet path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--llm", type=str, default="qwen3-8b", help="LLM tag")
    parser.add_argument("--layer", type=int, default=-2, help="Layer index")
    parser.add_argument("--shard-size", type=int, default=1000, help="Shard size")
    parser.add_argument("--max-samples", type=int, help="Max samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # LLM config mapping
    llm_configs = {
        "qwen3-8b": {
            "model_name": "Qwen/Qwen3-8B-Instruct",
            "quantization": "awq-4bit",
            "dtype": "bfloat16",
        },
        "qwen3-32b": {
            "model_name": "Qwen/Qwen3-32B-Instruct",
            "quantization": "awq-4bit",
            "dtype": "bfloat16",
        },
    }

    llm_config = llm_configs.get(args.llm, llm_configs["qwen3-8b"])

    summary = build_latents_from_cots(
        cots_path=args.cots,
        output_dir=args.output,
        llm_config=llm_config,
        layer_idx=args.layer,
        shard_size=args.shard_size,
        max_samples=args.max_samples,
        device=args.device,
    )

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
