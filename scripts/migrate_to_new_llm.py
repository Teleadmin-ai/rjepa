"""
CLI tool for migrating R-JEPA to a new LLM (Llama, Mistral, DeepSeek, etc.).

Usage:
    # Migrate from Qwen3-8B to Llama3-70B
    python scripts/migrate_to_new_llm.py \\
        --source qwen3-8b \\
        --target llama3-70b \\
        --strategy calibration

    # Migrate to Mistral-8x22B with full retrain
    python scripts/migrate_to_new_llm.py \\
        --source qwen3-8b \\
        --target mixtral-8x22b \\
        --strategy retrain
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rjepa.pipeline.calibrate import create_calibration_pipeline
from rjepa.llm.projections import LLM_HIDDEN_SIZES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/migrate_llm.log"),
    ],
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate R-JEPA to a new open-source LLM"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="qwen3-8b",
        help="Source LLM (base R-JEPA trained on this)",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target LLM (e.g., llama3-70b, mistral-7b, deepseek-67b)",
    )

    parser.add_argument(
        "--source-checkpoint",
        type=str,
        default="data/checkpoints/rjepa-qwen3-8b/latest.pth",
        help="Source R-JEPA checkpoint",
    )

    parser.add_argument(
        "--calibration-samples",
        type=str,
        default=None,
        help="Path to calibration latents (pre-generated)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of calibration samples",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["calibration", "transfer", "retrain"],
        default="calibration",
        help="Migration strategy:\n"
        "  calibration: Fast (2-4h), train projections only\n"
        "  transfer: Medium (12-24h), transfer weights + fine-tune\n"
        "  retrain: Slow (2-3 days), full retrain on new LLM",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for adapter/checkpoint",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("R-JEPA MULTI-LLM MIGRATION")
    logger.info("=" * 80)
    logger.info(f"Source LLM: {args.source}")
    logger.info(f"Target LLM: {args.target}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Source checkpoint: {args.source_checkpoint}")
    logger.info(f"Calibration samples: {args.num_samples}")
    logger.info("=" * 80)

    # Check if target LLM is supported
    if args.target not in LLM_HIDDEN_SIZES:
        logger.warning(
            f"Target LLM '{args.target}' not in reference list. "
            f"Supported: {list(LLM_HIDDEN_SIZES.keys())}\n"
            f"Will attempt auto-detection from model config."
        )

    # Execute strategy
    if args.strategy == "calibration":
        logger.info("\n[STRATEGY: CALIBRATION]")
        logger.info("  1. Load base R-JEPA (frozen)")
        logger.info("  2. Create adapter W_in/W_out")
        logger.info("  3. Train adapter on calibration set (~5k samples)")
        logger.info("  4. Save calibrated adapter")
        logger.info("  Expected time: 2-4 hours\n")

        # Create pipeline
        pipeline = create_calibration_pipeline(
            base_rjepa_checkpoint=args.source_checkpoint,
            base_llm_tag=args.source,
        )

        # Calibrate
        try:
            adapter_path, loss = pipeline.calibrate_for_llm(
                target_llm_tag=args.target,
                calibration_samples_path=(
                    Path(args.calibration_samples) if args.calibration_samples else None
                ),
                num_calibration_samples=args.num_samples,
                output_dir=Path(args.output_dir) if args.output_dir else None,
            )

            logger.info("\n" + "=" * 80)
            logger.info("MIGRATION SUCCESSFUL")
            logger.info("=" * 80)
            logger.info(f"Adapter saved: {adapter_path}")
            logger.info(f"Calibration loss: {loss:.4f}")
            logger.info("\nNext steps:")
            logger.info(f"  1. Evaluate on benchmark:")
            logger.info(f"     python -m rjepa.pipeline.evaluate \\")
            logger.info(f"       --llm {args.target} \\")
            logger.info(f"       --adapter {adapter_path} \\")
            logger.info(f"       --bench gsm8k")
            logger.info(f"  2. If performance good, deploy to production")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"\nMIGRATION FAILED: {e}", exc_info=True)
            sys.exit(1)

    elif args.strategy == "transfer":
        logger.info("\n[STRATEGY: TRANSFER LEARNING]")
        logger.info("  1. Load base R-JEPA weights")
        logger.info("  2. Upsample matrices to target size")
        logger.info("  3. Fine-tune on 20% of dataset")
        logger.info("  4. Save transferred checkpoint")
        logger.info("  Expected time: 12-24 hours\n")

        logger.error("Transfer learning strategy not yet implemented!")
        logger.info("Use --strategy calibration for now.")
        sys.exit(1)

    elif args.strategy == "retrain":
        logger.info("\n[STRATEGY: FULL RETRAIN]")
        logger.info("  1. Regenerate latents with new LLM")
        logger.info("  2. Train R-JEPA from scratch (same config)")
        logger.info("  3. Save new checkpoint")
        logger.info("  Expected time: 2-3 days\n")

        logger.error("Full retrain strategy not yet implemented!")
        logger.info("Use --strategy calibration for faster migration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
