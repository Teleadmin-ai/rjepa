"""
Test R-JEPA Trainer.
"""
import pytest
import torch
import tempfile
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from rjepa.jepa import ReasoningJEPA, ContiguousMasker, MaskCollator
from rjepa.jepa.trainer import RJEPATrainer


@pytest.fixture
def dummy_model():
    """Create small dummy model for testing."""
    return ReasoningJEPA(
        dim=64,
        depth_encoder=2,
        depth_predictor=2,
        num_heads=2,
        predictor_dim=128,
        max_steps=32,
        domain_embed_dim=16,
        num_domains=5,
        ema_momentum=0.9,
        loss_config={"loss_type": "l1", "var_reg_weight": 0.01},
    )


@pytest.fixture
def dummy_dataloader():
    """Create dummy dataloader with masking."""
    # Create dummy latent data
    latents = torch.randn(20, 10, 64)  # 20 samples, 10 steps, 64 dim
    domain_ids = torch.randint(0, 5, (20,))

    # Wrap in dataset
    dataset = TensorDataset(latents, domain_ids)

    # Create masker and collator
    masker = ContiguousMasker(min_mask_ratio=0.3, max_mask_ratio=0.5)

    def collate_fn(batch):
        latents_batch = torch.stack([item[0] for item in batch])
        domain_ids_batch = torch.stack([item[1] for item in batch])

        batch_size, num_steps, dim = latents_batch.shape

        # Generate masks
        context_mask, target_mask = masker(batch_size, num_steps)

        return {
            "latents": latents_batch,
            "domain_ids": domain_ids_batch,
            "context_mask": context_mask,
            "target_mask": target_mask,
        }

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return loader


def test_trainer_initialization(dummy_model, dummy_dataloader):
    """Test trainer initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = RJEPATrainer(
            model=dummy_model,
            train_loader=dummy_dataloader,
            val_loader=None,
            lr=1e-3,
            max_epochs=5,
            checkpoint_dir=Path(tmpdir),
            use_wandb=False,
            device="cpu",
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0


def test_trainer_single_epoch(dummy_model, dummy_dataloader):
    """Test training one epoch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = RJEPATrainer(
            model=dummy_model,
            train_loader=dummy_dataloader,
            val_loader=None,
            lr=1e-3,
            max_epochs=1,
            checkpoint_dir=Path(tmpdir),
            use_wandb=False,
            amp_enabled=False,  # Disable AMP for CPU
            device="cpu",
        )

        # Train one epoch
        epoch_metrics = trainer.train_epoch()

        assert "loss" in epoch_metrics
        assert "recon_loss" in epoch_metrics
        assert "var_reg_loss" in epoch_metrics

        assert epoch_metrics["loss"] > 0
        assert trainer.global_step > 0


def test_trainer_validation(dummy_model, dummy_dataloader):
    """Test validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = RJEPATrainer(
            model=dummy_model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,  # Use train as val for testing
            lr=1e-3,
            max_epochs=1,
            checkpoint_dir=Path(tmpdir),
            use_wandb=False,
            amp_enabled=False,
            device="cpu",
        )

        # Run validation
        val_metrics = trainer.validate()

        assert "loss" in val_metrics
        assert "recon_loss" in val_metrics
        assert val_metrics["loss"] > 0


def test_trainer_checkpointing(dummy_model, dummy_dataloader):
    """Test checkpoint saving and loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # Create trainer
        trainer = RJEPATrainer(
            model=dummy_model,
            train_loader=dummy_dataloader,
            val_loader=None,
            lr=1e-3,
            max_epochs=1,
            checkpoint_dir=checkpoint_dir,
            use_wandb=False,
            amp_enabled=False,
            device="cpu",
        )

        # Train one epoch
        trainer.train_epoch()
        trainer.current_epoch = 1

        # Save checkpoint
        trainer.save_checkpoint(filename="test_checkpoint.pth")

        checkpoint_path = checkpoint_dir / "test_checkpoint.pth"
        assert checkpoint_path.exists()

        # Create new trainer and load checkpoint
        new_trainer = RJEPATrainer(
            model=ReasoningJEPA(
                dim=64,
                depth_encoder=2,
                depth_predictor=2,
                num_heads=2,
            ),
            train_loader=dummy_dataloader,
            val_loader=None,
            lr=1e-3,
            max_epochs=10,
            checkpoint_dir=checkpoint_dir,
            use_wandb=False,
            device="cpu",
        )

        new_trainer.load_checkpoint(checkpoint_path)

        assert new_trainer.current_epoch == 1
        assert new_trainer.global_step == trainer.global_step


def test_trainer_full_training(dummy_model, dummy_dataloader):
    """Test full training loop (2 epochs)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = RJEPATrainer(
            model=dummy_model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            lr=1e-3,
            max_epochs=2,
            checkpoint_dir=Path(tmpdir),
            use_wandb=False,
            amp_enabled=False,
            device="cpu",
        )

        # Run full training
        history = trainer.train()

        assert "best_val_loss" in history
        assert "total_epochs" in history
        assert "total_time" in history

        assert history["total_epochs"] == 2

        # Check checkpoints were saved
        checkpoint_dir = Path(tmpdir)
        assert (checkpoint_dir / "final.pth").exists()


def test_ema_momentum_annealing(dummy_model, dummy_dataloader):
    """Test EMA momentum annealing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = RJEPATrainer(
            model=dummy_model,
            train_loader=dummy_dataloader,
            val_loader=None,
            lr=1e-3,
            max_epochs=10,
            ema_momentum_start=0.9,
            ema_momentum_end=0.99,
            checkpoint_dir=Path(tmpdir),
            use_wandb=False,
            device="cpu",
        )

        # Get momentum at different epochs
        trainer.current_epoch = 0
        momentum_start = trainer._get_ema_momentum()

        trainer.current_epoch = 5
        momentum_mid = trainer._get_ema_momentum()

        trainer.current_epoch = 9
        momentum_end = trainer._get_ema_momentum()

        # Should increase over time
        assert momentum_start < momentum_mid < momentum_end
        assert abs(momentum_start - 0.9) < 0.01
        assert abs(momentum_end - 0.99) < 0.01


def test_lr_scheduler(dummy_model, dummy_dataloader):
    """Test learning rate scheduling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = RJEPATrainer(
            model=dummy_model,
            train_loader=dummy_dataloader,
            val_loader=None,
            lr=1e-3,
            max_epochs=10,
            warmup_epochs=2,
            checkpoint_dir=Path(tmpdir),
            use_wandb=False,
            amp_enabled=False,
            device="cpu",
        )

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # Train one epoch (should be in warmup)
        trainer.train_epoch()

        lr_after_epoch1 = trainer.optimizer.param_groups[0]["lr"]

        # LR should increase during warmup
        assert lr_after_epoch1 > initial_lr or abs(lr_after_epoch1 - initial_lr) < 1e-6
