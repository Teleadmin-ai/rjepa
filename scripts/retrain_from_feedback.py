"""
CLI tool for manual retraining from user feedback.

Usage:
    python scripts/retrain_from_feedback.py --days 7 --deploy
    python scripts/retrain_from_feedback.py --version user-feedback-20250117 --no-deploy
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rjepa.pipeline.continuous_learning import create_continuous_learning_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/retrain_from_feedback.log"),
    ],
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Retrain R-JEPA from user feedback (manual trigger)"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back for feedback (default: 7)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/checkpoints/rjepa-qwen3-8b/latest.pth",
        help="Base checkpoint to fine-tune from",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/interactions",
        help="Directory with interaction logs",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum new samples to trigger retraining",
    )

    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Automatically deploy if A/B test passes",
    )

    parser.add_argument(
        "--no-deploy",
        dest="deploy",
        action="store_false",
        help="Do not deploy, only evaluate",
    )

    parser.set_defaults(deploy=False)

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MANUAL RETRAINING FROM USER FEEDBACK")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Days back: {args.days}")
    logger.info(f"  Base checkpoint: {args.checkpoint}")
    logger.info(f"  Log directory: {args.log_dir}")
    logger.info(f"  Min samples: {args.min_samples}")
    logger.info(f"  Auto-deploy: {args.deploy}")
    logger.info("=" * 80)

    # Create pipeline
    pipeline = create_continuous_learning_pipeline(
        base_checkpoint=args.checkpoint,
        log_dir=args.log_dir,
        min_new_samples=args.min_samples,
    )

    # Run cycle
    result = pipeline.run_nightly_cycle(
        days_back=args.days,
        auto_deploy=args.deploy,
    )

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("RETRAINING SUMMARY")
    logger.info("=" * 80)

    if result["status"] == "success":
        logger.info(f"Status: SUCCESS")
        logger.info(f"Dataset version: {result['dataset_version']}")
        logger.info(f"New checkpoint: {result['new_checkpoint']}")
        logger.info(f"Deployed: {result['deployed']}")

        if result["deployed"]:
            logger.info("")
            logger.info("New checkpoint deployed to production!")
        else:
            logger.info("")
            logger.info("New checkpoint created but NOT deployed.")
            logger.info("Review metrics and deploy manually if desired.")

    elif result["status"] == "skipped":
        logger.info(f"Status: SKIPPED")
        logger.info(f"Reason: {result['reason']}")
        logger.info("")
        logger.info(f"Not enough new data to trigger retraining.")
        logger.info(f"Try again later or reduce --min-samples.")

    elif result["status"] == "failed":
        logger.error(f"Status: FAILED")
        logger.error(f"Error: {result['error']}")
        logger.error("")
        logger.error("Retraining failed. Check logs for details.")
        sys.exit(1)

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
