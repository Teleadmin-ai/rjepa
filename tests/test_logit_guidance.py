"""
Tests for Logit Guidance.
"""
import pytest
import torch

from rjepa.inference import (
    LogitGuidance,
    LogitGuidanceConfig,
    create_logit_guidance,
    LogitGuidanceTrainer,
)


# ============================================================================
# Test Model
# ============================================================================

def test_logit_guidance_config():
    """Test LogitGuidanceConfig dataclass."""
    config = LogitGuidanceConfig(
        latent_dim=4096,
        vocab_size=151936,
        hidden_dim=2048,
        alpha=0.3,
    )

    assert config.latent_dim == 4096
    assert config.vocab_size == 151936
    assert config.hidden_dim == 2048
    assert config.alpha == 0.3
    assert config.dropout == 0.1  # default
    assert config.temperature == 1.0  # default


def test_logit_guidance_initialization():
    """Test LogitGuidance initialization."""
    config = LogitGuidanceConfig(
        latent_dim=512,
        vocab_size=1000,
        hidden_dim=256,
    )

    guidance = LogitGuidance(config)

    # Check MLP exists
    assert hasattr(guidance, "mlp")

    # Check parameter count
    num_params = sum(p.numel() for p in guidance.parameters())
    assert num_params > 0


def test_logit_guidance_forward():
    """Test LogitGuidance forward pass."""
    config = LogitGuidanceConfig(
        latent_dim=512,
        vocab_size=1000,
        hidden_dim=256,
    )

    guidance = LogitGuidance(config)

    # Test with batch
    batch_size = 4
    latent = torch.randn(batch_size, config.latent_dim)

    logit_bias = guidance(latent)

    assert logit_bias.shape == (batch_size, config.vocab_size)

    # Test with single sample (should add batch dim)
    latent_single = torch.randn(config.latent_dim)
    logit_bias_single = guidance(latent_single)

    assert logit_bias_single.shape == (1, config.vocab_size)


def test_apply_guidance():
    """Test apply_guidance method."""
    config = LogitGuidanceConfig(
        latent_dim=512,
        vocab_size=1000,
        hidden_dim=256,
        alpha=0.3,
    )

    guidance = LogitGuidance(config)

    batch_size = 4
    latent = torch.randn(batch_size, config.latent_dim)
    llm_logits = torch.randn(batch_size, config.vocab_size)

    # Apply guidance
    guided_logits = guidance.apply_guidance(
        llm_logits=llm_logits,
        latent=latent,
    )

    # Check shape
    assert guided_logits.shape == llm_logits.shape

    # Check that guidance was applied (should be different)
    assert not torch.allclose(guided_logits, llm_logits)


def test_apply_guidance_alpha_zero():
    """Test that alpha=0 returns original logits."""
    config = LogitGuidanceConfig(
        latent_dim=512,
        vocab_size=1000,
        hidden_dim=256,
    )

    guidance = LogitGuidance(config)

    batch_size = 4
    latent = torch.randn(batch_size, config.latent_dim)
    llm_logits = torch.randn(batch_size, config.vocab_size)

    # Apply guidance with alpha=0 (no guidance)
    guided_logits = guidance.apply_guidance(
        llm_logits=llm_logits,
        latent=latent,
        alpha=0.0,
    )

    # Should be identical to original logits
    assert torch.allclose(guided_logits, llm_logits, atol=1e-5)


def test_apply_guidance_custom_alpha():
    """Test guidance with custom alpha value."""
    config = LogitGuidanceConfig(
        latent_dim=512,
        vocab_size=1000,
        hidden_dim=256,
        alpha=0.3,  # default
    )

    guidance = LogitGuidance(config)

    batch_size = 4
    latent = torch.randn(batch_size, config.latent_dim)
    llm_logits = torch.randn(batch_size, config.vocab_size)

    # Apply with different alphas
    guided_low = guidance.apply_guidance(llm_logits, latent, alpha=0.1)
    guided_high = guidance.apply_guidance(llm_logits, latent, alpha=0.9)

    # High alpha should have larger delta from original
    delta_low = (guided_low - llm_logits).abs().mean()
    delta_high = (guided_high - llm_logits).abs().mean()

    assert delta_high > delta_low


def test_create_logit_guidance():
    """Test create_logit_guidance factory function."""
    guidance = create_logit_guidance(
        latent_dim=512,
        vocab_size=1000,
        hidden_dim=256,
        alpha=0.3,
    )

    assert isinstance(guidance, LogitGuidance)
    assert guidance.config.latent_dim == 512
    assert guidance.config.vocab_size == 1000
    assert guidance.config.alpha == 0.3


# ============================================================================
# Test Trainer
# ============================================================================

def test_trainer_initialization():
    """Test LogitGuidanceTrainer initialization."""
    from torch.utils.data import DataLoader

    config = LogitGuidanceConfig(
        latent_dim=512,
        vocab_size=1000,
        hidden_dim=256,
    )

    guidance = LogitGuidance(config)

    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "latent": torch.randn(512),
                "llm_logits": torch.randn(1000),
                "true_token": torch.randint(0, 1000, ()),
            }

    dataset = DummyDataset(50)
    loader = DataLoader(dataset, batch_size=8)

    # Create trainer
    trainer = LogitGuidanceTrainer(
        guidance=guidance,
        train_loader=loader,
        lr=3e-4,
        device="cpu",
        use_amp=False,
        wandb_enabled=False,
    )

    assert trainer.guidance is guidance
    assert trainer.device == "cpu"
    assert trainer.use_amp is False
    assert trainer.optimizer is not None
    assert trainer.scaler is None  # No scaler when AMP disabled


def test_trainer_single_epoch():
    """Test training for one epoch."""
    from torch.utils.data import DataLoader

    config = LogitGuidanceConfig(
        latent_dim=512,
        vocab_size=1000,
        hidden_dim=256,
    )

    guidance = LogitGuidance(config)

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "latent": torch.randn(512),
                "llm_logits": torch.randn(1000),
                "true_token": torch.randint(0, 1000, ()),
            }

    dataset = DummyDataset(50)
    loader = DataLoader(dataset, batch_size=8)

    trainer = LogitGuidanceTrainer(
        guidance=guidance,
        train_loader=loader,
        lr=3e-4,
        device="cpu",
        use_amp=False,
        wandb_enabled=False,
    )

    # Train one epoch
    metrics = trainer.train_epoch()

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] >= 0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_trainer_validation():
    """Test validation loop."""
    from torch.utils.data import DataLoader

    config = LogitGuidanceConfig(
        latent_dim=512,
        vocab_size=1000,
        hidden_dim=256,
    )

    guidance = LogitGuidance(config)

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "latent": torch.randn(512),
                "llm_logits": torch.randn(1000),
                "true_token": torch.randint(0, 1000, ()),
            }

    train_data = DummyDataset(50)
    val_data = DummyDataset(20)

    train_loader = DataLoader(train_data, batch_size=8)
    val_loader = DataLoader(val_data, batch_size=8)

    trainer = LogitGuidanceTrainer(
        guidance=guidance,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=3e-4,
        device="cpu",
        use_amp=False,
        wandb_enabled=False,
    )

    # Validate
    val_metrics = trainer.validate()

    assert "loss" in val_metrics
    assert "accuracy" in val_metrics
    assert val_metrics["loss"] >= 0


def test_trainer_checkpointing():
    """Test checkpoint save/load."""
    import tempfile
    from pathlib import Path
    from torch.utils.data import DataLoader

    config = LogitGuidanceConfig(
        latent_dim=512,
        vocab_size=1000,
        hidden_dim=256,
    )

    guidance = LogitGuidance(config)

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "latent": torch.randn(512),
                "llm_logits": torch.randn(1000),
                "true_token": torch.randint(0, 1000, ()),
            }

    dataset = DummyDataset(50)
    loader = DataLoader(dataset, batch_size=8)

    trainer = LogitGuidanceTrainer(
        guidance=guidance,
        train_loader=loader,
        lr=3e-4,
        device="cpu",
        use_amp=False,
        wandb_enabled=False,
    )

    # Train one step to update state
    trainer.train_epoch()

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # Create new trainer and load checkpoint
        new_guidance = LogitGuidance(config)
        new_trainer = LogitGuidanceTrainer(
            guidance=new_guidance,
            train_loader=loader,
            lr=3e-4,
            device="cpu",
            use_amp=False,
            wandb_enabled=False,
        )

        new_trainer.load_checkpoint(checkpoint_path)

        # Check state was restored
        assert new_trainer.global_step == trainer.global_step
        assert new_trainer.epoch == trainer.epoch


def test_trainer_full_training():
    """Test full training for 2 epochs."""
    import tempfile
    from pathlib import Path
    from torch.utils.data import DataLoader

    config = LogitGuidanceConfig(
        latent_dim=512,
        vocab_size=1000,
        hidden_dim=256,
    )

    guidance = LogitGuidance(config)

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "latent": torch.randn(512),
                "llm_logits": torch.randn(1000),
                "true_token": torch.randint(0, 1000, ()),
            }

    dataset = DummyDataset(50)
    loader = DataLoader(dataset, batch_size=8)

    trainer = LogitGuidanceTrainer(
        guidance=guidance,
        train_loader=loader,
        lr=3e-4,
        device="cpu",
        use_amp=False,
        wandb_enabled=False,
    )

    # Train for 2 epochs
    with tempfile.TemporaryDirectory() as tmpdir:
        history = trainer.train(
            num_epochs=2,
            save_dir=Path(tmpdir),
            save_interval=1,
        )

        # Check history
        assert "train_loss" in history
        assert "train_accuracy" in history
        assert len(history["train_loss"]) == 2
        assert len(history["train_accuracy"]) == 2

        # Check checkpoints were saved
        checkpoints = list(Path(tmpdir).glob("checkpoint-epoch-*.pth"))
        assert len(checkpoints) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
