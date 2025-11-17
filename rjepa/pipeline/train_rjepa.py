"""
Training Pipeline for R-JEPA.

Orchestrates end-to-end training:
1. Load latent datasets
2. Create dataloaders with masking
3. Initialize model
4. Train with checkpointing + W&B
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
import yaml

from rjepa.jepa import (
    ReasoningJEPA,
    LatentDataset,
    ContiguousMasker,
    MaskCollator,
    create_rjepa_model,
)
from rjepa.jepa.trainer import RJEPATrainer

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(
    train_latents_dir: Path,
    val_latents_dir: Optional[Path],
    batch_size: int,
    num_workers: int,
    masker_config: Dict,
    device: str = "cpu",
) -> tuple:
    """
    Create train and validation dataloaders.

    Args:
        train_latents_dir: Training latents directory
        val_latents_dir: Optional validation latents directory
        batch_size: Batch size
        num_workers: DataLoader num_workers
        masker_config: Masker configuration dict
        device: Device for masking

    Returns:
        (train_loader, val_loader)
    """
    logger.info("Creating dataloaders...")

    # Create datasets
    train_dataset = LatentDataset(train_latents_dir, device="cpu")
    logger.info(f"Train dataset: {len(train_dataset)} samples")

    val_dataset = None
    if val_latents_dir and val_latents_dir.exists():
        val_dataset = LatentDataset(val_latents_dir, device="cpu")
        logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create masker
    from rjepa.jepa.maskers import create_masker
    masker = create_masker(masker_config)

    # Create collator
    collator = MaskCollator(masker, device=device)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=(device == "cuda"),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=(device == "cuda"),
        )

    logger.info(f"Train loader: {len(train_loader)} batches")
    if val_loader:
        logger.info(f"Val loader: {len(val_loader)} batches")

    return train_loader, val_loader


def train_rjepa_from_config(
    config_path: Path,
    output_dir: Path,
    resume_from: Optional[Path] = None,
):
    """
    Train R-JEPA from config file.

    Args:
        config_path: Path to config YAML
        output_dir: Output directory for checkpoints
        resume_from: Optional checkpoint to resume from
    """
    logger.info("=" * 80)
    logger.info("R-JEPA TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")

    # Load config
    config = load_config(config_path)

    # Create output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output dir
    import shutil
    shutil.copy(config_path, output_dir / "config.yaml")

    # Device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_latents_dir=Path(config["data"]["train_latents_dir"]),
        val_latents_dir=Path(config["data"].get("val_latents_dir"))
        if "val_latents_dir" in config["data"]
        else None,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 4),
        masker_config=config.get("masker", {"type": "contiguous"}),
        device=device,
    )

    # Create model
    logger.info("Creating R-JEPA model...")
    model = create_rjepa_model(config["model"])
    logger.info(
        f"Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params"
    )

    # Create trainer
    logger.info("Creating trainer...")

    # W&B config
    wandb_config = None
    if config.get("wandb", {}).get("enabled", False):
        wandb_config = {
            "project": config["wandb"].get("project", "rjepa-training"),
            "name": config["wandb"].get("run_name"),
            "config": config,
        }

    trainer = RJEPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.05),
        warmup_epochs=config["training"].get("warmup_epochs", 10),
        max_epochs=config["training"]["max_epochs"],
        grad_clip=config["training"].get("grad_clip", 1.0),
        amp_enabled=config["training"].get("amp_enabled", True),
        ema_momentum_start=config["training"].get("ema_momentum_start", 0.996),
        ema_momentum_end=config["training"].get("ema_momentum_end", 0.9999),
        checkpoint_dir=output_dir,
        log_interval=config["training"].get("log_interval", 10),
        val_interval=config["training"].get("val_interval", 1),
        use_wandb=config.get("wandb", {}).get("enabled", False),
        wandb_config=wandb_config,
        device=device,
    )

    # Resume if requested
    if resume_from:
        trainer.load_checkpoint(resume_from)

    # Train
    history = trainer.train()

    # Save final summary
    import json

    summary = {
        "config": config,
        "history": history,
        "best_val_loss": history["best_val_loss"],
        "total_epochs": history["total_epochs"],
        "total_time_hours": history["total_time"] / 3600,
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training summary saved to {summary_path}")

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# PREFECT FLOW
# ═════════════════════════════════════════════════════════════════════════════

try:
    from prefect import flow, task

    @task(name="load_config_task")
    def load_config_task(config_path: Path):
        """Load config as Prefect task."""
        return load_config(config_path)

    @task(name="create_dataloaders_task")
    def create_dataloaders_task(config: Dict, device: str):
        """Create dataloaders as Prefect task."""
        return create_dataloaders(
            train_latents_dir=Path(config["data"]["train_latents_dir"]),
            val_latents_dir=Path(config["data"].get("val_latents_dir"))
            if "val_latents_dir" in config["data"]
            else None,
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"].get("num_workers", 4),
            masker_config=config.get("masker", {"type": "contiguous"}),
            device=device,
        )

    @task(name="create_model_task")
    def create_model_task(config: Dict):
        """Create model as Prefect task."""
        return create_rjepa_model(config["model"])

    @task(name="train_task")
    def train_task(
        model,
        train_loader,
        val_loader,
        config: Dict,
        output_dir: Path,
        device: str,
    ):
        """Training as Prefect task."""
        # W&B config
        wandb_config = None
        if config.get("wandb", {}).get("enabled", False):
            wandb_config = {
                "project": config["wandb"].get("project", "rjepa-training"),
                "name": config["wandb"].get("run_name"),
                "config": config,
            }

        trainer = RJEPATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=config["training"]["lr"],
            weight_decay=config["training"].get("weight_decay", 0.05),
            warmup_epochs=config["training"].get("warmup_epochs", 10),
            max_epochs=config["training"]["max_epochs"],
            grad_clip=config["training"].get("grad_clip", 1.0),
            amp_enabled=config["training"].get("amp_enabled", True),
            ema_momentum_start=config["training"].get("ema_momentum_start", 0.996),
            ema_momentum_end=config["training"].get("ema_momentum_end", 0.9999),
            checkpoint_dir=output_dir,
            log_interval=config["training"].get("log_interval", 10),
            val_interval=config["training"].get("val_interval", 1),
            use_wandb=config.get("wandb", {}).get("enabled", False),
            wandb_config=wandb_config,
            device=device,
        )

        return trainer.train()

    @flow(name="train_rjepa_flow")
    def train_rjepa_flow(
        config_path: str,
        output_dir: str,
        resume_from: Optional[str] = None,
    ):
        """
        Prefect flow for R-JEPA training.

        Args:
            config_path: Path to config YAML
            output_dir: Output directory
            resume_from: Optional checkpoint to resume from
        """
        config_path = Path(config_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        config = load_config_task(config_path)

        # Device
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Create dataloaders
        train_loader, val_loader = create_dataloaders_task(config, device)

        # Create model
        model = create_model_task(config)

        # Train
        history = train_task(model, train_loader, val_loader, config, output_dir, device)

        # Save summary
        import json

        summary = {
            "config": config,
            "history": history,
            "best_val_loss": history["best_val_loss"],
            "total_epochs": history["total_epochs"],
            "total_time_hours": history["total_time"] / 3600,
        }

        summary_path = output_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

except ImportError:
    logger.warning("Prefect not installed, flows not available")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train R-JEPA model")
    parser.add_argument(
        "--config", type=Path, required=True, help="Config YAML path"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--resume", type=Path, help="Checkpoint to resume from"
    )

    args = parser.parse_args()

    summary = train_rjepa_from_config(
        config_path=args.config,
        output_dir=args.output,
        resume_from=args.resume,
    )

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best validation loss: {summary['history']['best_val_loss']:.4f}")
    print(f"Total epochs: {summary['history']['total_epochs']}")
    print(f"Total time: {summary['history']['total_time'] / 3600:.2f} hours")
    print("=" * 80)
