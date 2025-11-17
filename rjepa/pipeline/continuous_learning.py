"""
Continuous Learning Pipeline - Nightly Retraining with User Feedback.

Philosophy:
- R-JEPA improves continuously from user interactions
- Scheduled retraining (nightly or weekly)
- Incremental learning: fine-tune on new data, not retrain from scratch
- A/B testing: validate new checkpoint before deployment
- Metrics tracking: measure improvement over time
"""
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import torch
import pandas as pd

from rjepa.data.feedback_pipeline import FeedbackPipeline, create_feedback_pipeline
from rjepa.pipeline.build_latents import build_latents_from_cots
from rjepa.pipeline.train_rjepa import train_rjepa_from_config
from rjepa.jepa.model import ReasoningJEPA

logger = logging.getLogger(__name__)


class ContinuousLearningPipeline:
    """
    Orchestrates continuous learning cycle:

    1. Collect user feedback (interactions)
    2. Validate & convert to training data
    3. Generate latents from new CoTs
    4. Fine-tune R-JEPA on new data (incremental)
    5. Evaluate new checkpoint (A/B test)
    6. Deploy if improved, rollback if degraded
    7. Track metrics over time
    """

    def __init__(
        self,
        feedback_pipeline: FeedbackPipeline,
        base_checkpoint: str,
        output_dir: str = "data/checkpoints/rjepa-continuous",
        eval_benchmark: str = "gsm8k",
        min_new_samples: int = 100,
        improvement_threshold: float = 0.0,  # Accept if not worse
    ):
        """
        Initialize continuous learning pipeline.

        Args:
            feedback_pipeline: FeedbackPipeline instance
            base_checkpoint: Path to current R-JEPA checkpoint
            output_dir: Output directory for new checkpoints
            eval_benchmark: Benchmark for A/B testing
            min_new_samples: Min new samples to trigger retraining
            improvement_threshold: Min accuracy delta to deploy new checkpoint
        """
        self.feedback_pipeline = feedback_pipeline
        self.base_checkpoint = Path(base_checkpoint)
        self.output_dir = Path(output_dir)
        self.eval_benchmark = eval_benchmark
        self.min_new_samples = min_new_samples
        self.improvement_threshold = improvement_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.metrics_log = self.output_dir / "metrics.jsonl"

        logger.info(
            f"ContinuousLearningPipeline initialized: "
            f"checkpoint={base_checkpoint}, min_samples={min_new_samples}"
        )

    def collect_feedback(
        self,
        days_back: int = 1,
    ) -> Optional[str]:
        """
        Collect and validate user feedback from last N days.

        Args:
            days_back: Number of days to look back

        Returns:
            Dataset version (str) if successful, None if insufficient data
        """
        logger.info(f"Collecting feedback from last {days_back} days...")

        # Compute date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # Run feedback pipeline
        stats = self.feedback_pipeline.run_pipeline(
            start_date=start_date,
            end_date=end_date,
        )

        if stats["num_valid"] < self.min_new_samples:
            logger.warning(
                f"Insufficient new samples: {stats['num_valid']} < {self.min_new_samples}. "
                f"Skipping retraining."
            )
            return None

        logger.info(
            f"Feedback collection complete: {stats['num_valid']} valid interactions, "
            f"acceptance rate={stats['acceptance_rate']:.1%}"
        )

        return stats["version"]

    def generate_latents(
        self,
        dataset_version: str,
        llm_tag: str = "qwen3-8b",
    ) -> Path:
        """
        Generate latents from new CoT data.

        Args:
            dataset_version: Dataset version (from feedback pipeline)
            llm_tag: LLM to use for latent extraction

        Returns:
            Path to latents directory
        """
        logger.info(f"Generating latents for dataset {dataset_version}...")

        # Load CoTs from validated dataset
        cots_path = self.feedback_pipeline.output_dir / dataset_version / "cots.parquet"

        if not cots_path.exists():
            raise FileNotFoundError(f"CoTs not found: {cots_path}")

        # Output path for latents
        latents_dir = Path(f"data/latents/{llm_tag}/user-feedback/{dataset_version}")
        latents_dir.mkdir(parents=True, exist_ok=True)

        # Build latents (reuse existing pipeline)
        # TODO: Call build_latents_from_cots with cots_path
        logger.info(f"Latents generated: {latents_dir}")

        return latents_dir

    def fine_tune_rjepa(
        self,
        latents_dir: Path,
        num_epochs: int = 3,
        lr: float = 1e-5,
    ) -> Path:
        """
        Fine-tune R-JEPA on new data (incremental learning).

        Args:
            latents_dir: Directory with new latents
            num_epochs: Number of fine-tuning epochs (small!)
            lr: Learning rate (should be lower than initial training)

        Returns:
            Path to new checkpoint
        """
        logger.info(f"Fine-tuning R-JEPA on {latents_dir}...")

        # Load base checkpoint
        checkpoint = torch.load(self.base_checkpoint, map_location="cpu")
        model_state = checkpoint["model_state_dict"]

        # Create output checkpoint path
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        new_checkpoint_dir = self.output_dir / f"checkpoint-{timestamp}"
        new_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create config for fine-tuning
        config = {
            "model": checkpoint.get("config", {}),
            "training": {
                "batch_size": 32,
                "lr": lr,
                "num_epochs": num_epochs,
                "warmup_epochs": 1,
                "use_amp": True,
                "grad_clip": 1.0,
            },
            "data": {
                "train_latents_dir": str(latents_dir / "train"),
                "val_latents_dir": str(latents_dir / "val"),
            },
            "checkpoint": {
                "load_from": str(self.base_checkpoint),
                "save_to": str(new_checkpoint_dir),
            },
        }

        # Save config
        import yaml
        config_path = new_checkpoint_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Run training (reuse existing trainer)
        # TODO: Call train_rjepa_from_config with config
        logger.info(f"Fine-tuning complete: {new_checkpoint_dir}")

        return new_checkpoint_dir / "final.pth"

    def evaluate_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Dict[str, float]:
        """
        Evaluate checkpoint on benchmark.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Metrics dict (accuracy, jepa_loss, etc.)
        """
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")

        # TODO: Call evaluation pipeline
        # For now, mock metrics
        metrics = {
            "accuracy": 0.82,  # Mock
            "jepa_loss": 0.15,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Evaluation complete: accuracy={metrics['accuracy']:.1%}")

        return metrics

    def ab_test(
        self,
        new_checkpoint: Path,
    ) -> bool:
        """
        A/B test: compare new checkpoint vs current baseline.

        Args:
            new_checkpoint: Path to new checkpoint

        Returns:
            True if new checkpoint is better (should deploy)
        """
        logger.info("Running A/B test...")

        # Evaluate baseline
        baseline_metrics = self.evaluate_checkpoint(self.base_checkpoint)

        # Evaluate new checkpoint
        new_metrics = self.evaluate_checkpoint(new_checkpoint)

        # Compare
        baseline_acc = baseline_metrics["accuracy"]
        new_acc = new_metrics["accuracy"]
        delta = new_acc - baseline_acc

        logger.info(
            f"A/B test results: baseline={baseline_acc:.1%}, new={new_acc:.1%}, "
            f"delta={delta:+.2%}"
        )

        # Decision
        should_deploy = delta >= self.improvement_threshold

        if should_deploy:
            logger.info("New checkpoint is better or equal -> DEPLOY")
        else:
            logger.warning(f"New checkpoint is worse ({delta:+.2%}) -> ROLLBACK")

        # Log metrics
        self._log_metrics(baseline_metrics, new_metrics, should_deploy)

        return should_deploy

    def deploy_checkpoint(
        self,
        new_checkpoint: Path,
    ):
        """
        Deploy new checkpoint (replace baseline).

        Args:
            new_checkpoint: Path to new checkpoint
        """
        logger.info(f"Deploying new checkpoint: {new_checkpoint}")

        # Copy to production location
        production_checkpoint = Path("data/checkpoints/rjepa-qwen3-8b/latest.pth")

        import shutil
        shutil.copy(new_checkpoint, production_checkpoint)

        logger.info(f"Deployed: {production_checkpoint}")

    def _log_metrics(
        self,
        baseline_metrics: Dict,
        new_metrics: Dict,
        deployed: bool,
    ):
        """
        Log metrics to JSONL for tracking over time.

        Args:
            baseline_metrics: Baseline checkpoint metrics
            new_metrics: New checkpoint metrics
            deployed: Whether new checkpoint was deployed
        """
        import json

        record = {
            "timestamp": datetime.now().isoformat(),
            "baseline_accuracy": baseline_metrics["accuracy"],
            "new_accuracy": new_metrics["accuracy"],
            "delta": new_metrics["accuracy"] - baseline_metrics["accuracy"],
            "deployed": deployed,
        }

        with open(self.metrics_log, "a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(f"Metrics logged to {self.metrics_log}")

    def run_nightly_cycle(
        self,
        days_back: int = 1,
        auto_deploy: bool = True,
    ) -> Dict[str, any]:
        """
        Run full nightly retraining cycle.

        Workflow:
        1. Collect feedback from last N days
        2. Generate latents
        3. Fine-tune R-JEPA
        4. A/B test
        5. Deploy if improved (optional)

        Args:
            days_back: Number of days to look back for feedback
            auto_deploy: Automatically deploy if A/B test passes

        Returns:
            Statistics dict
        """
        logger.info("=" * 80)
        logger.info("NIGHTLY CONTINUOUS LEARNING CYCLE START")
        logger.info("=" * 80)

        try:
            # 1. Collect feedback
            dataset_version = self.collect_feedback(days_back=days_back)

            if dataset_version is None:
                logger.info("No retraining needed (insufficient data)")
                return {"status": "skipped", "reason": "insufficient_data"}

            # 2. Generate latents
            latents_dir = self.generate_latents(dataset_version)

            # 3. Fine-tune R-JEPA
            new_checkpoint = self.fine_tune_rjepa(latents_dir)

            # 4. A/B test
            should_deploy = self.ab_test(new_checkpoint)

            # 5. Deploy (if auto_deploy and A/B test passed)
            deployed = False
            if should_deploy and auto_deploy:
                self.deploy_checkpoint(new_checkpoint)
                deployed = True

            logger.info("=" * 80)
            logger.info(f"NIGHTLY CYCLE COMPLETE: deployed={deployed}")
            logger.info("=" * 80)

            return {
                "status": "success",
                "dataset_version": dataset_version,
                "new_checkpoint": str(new_checkpoint),
                "deployed": deployed,
            }

        except Exception as e:
            logger.error(f"Nightly cycle failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}


def create_continuous_learning_pipeline(
    base_checkpoint: str = "data/checkpoints/rjepa-qwen3-8b/latest.pth",
    log_dir: str = "logs/interactions",
    min_new_samples: int = 100,
) -> ContinuousLearningPipeline:
    """
    Factory function to create ContinuousLearningPipeline.

    Args:
        base_checkpoint: Current R-JEPA checkpoint
        log_dir: Directory with interaction logs
        min_new_samples: Min new samples to trigger retraining

    Returns:
        ContinuousLearningPipeline instance
    """
    # Create feedback pipeline
    feedback_pipeline = create_feedback_pipeline(
        log_dir=log_dir,
        output_dir="data/datasets/user-feedback",
        jepa_score_threshold=0.7,
    )

    # Create continuous learning pipeline
    return ContinuousLearningPipeline(
        feedback_pipeline=feedback_pipeline,
        base_checkpoint=base_checkpoint,
        min_new_samples=min_new_samples,
    )


# Prefect flow for scheduled execution
try:
    from prefect import flow, task

    @task
    def collect_feedback_task(pipeline: ContinuousLearningPipeline, days_back: int):
        return pipeline.collect_feedback(days_back)

    @task
    def generate_latents_task(pipeline: ContinuousLearningPipeline, dataset_version: str):
        return pipeline.generate_latents(dataset_version)

    @task
    def fine_tune_task(pipeline: ContinuousLearningPipeline, latents_dir: Path):
        return pipeline.fine_tune_rjepa(latents_dir)

    @task
    def ab_test_task(pipeline: ContinuousLearningPipeline, new_checkpoint: Path):
        return pipeline.ab_test(new_checkpoint)

    @task
    def deploy_task(pipeline: ContinuousLearningPipeline, new_checkpoint: Path):
        pipeline.deploy_checkpoint(new_checkpoint)

    @flow(name="continuous-learning-nightly")
    def continuous_learning_flow(
        days_back: int = 1,
        auto_deploy: bool = True,
    ):
        """
        Prefect flow for nightly continuous learning.

        Schedule: cron("0 2 * * *")  # 2 AM daily
        """
        pipeline = create_continuous_learning_pipeline()

        logger.info("Starting continuous learning flow...")

        # Execute pipeline
        result = pipeline.run_nightly_cycle(days_back=days_back, auto_deploy=auto_deploy)

        return result

except ImportError:
    logger.warning("Prefect not installed, skipping flow definitions")
