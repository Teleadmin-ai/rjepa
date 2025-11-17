"""
Tests for Latent Decoder.
"""
import pytest
import torch
from transformers import AutoTokenizer

from rjepa.decoder import (
    LatentDecoder,
    LatentDecoderConfig,
    create_latent_decoder,
    LatentDecoderTrainer,
)


# ============================================================================
# Test Model
# ============================================================================

def test_latent_decoder_config():
    """Test LatentDecoderConfig dataclass."""
    config = LatentDecoderConfig(
        latent_dim=4096,
        vocab_size=151936,
        decoder_dim=1024,
        depth=4,
    )

    assert config.latent_dim == 4096
    assert config.vocab_size == 151936
    assert config.decoder_dim == 1024
    assert config.depth == 4
    assert config.num_heads == 8  # default
    assert config.tie_embeddings is True  # default


def test_latent_decoder_initialization():
    """Test LatentDecoder initialization."""
    config = LatentDecoderConfig(
        latent_dim=512,
        vocab_size=1000,
        decoder_dim=256,
        depth=2,
    )

    model = LatentDecoder(config)

    # Check components exist
    assert hasattr(model, "latent_proj")
    assert hasattr(model, "token_embed")
    assert hasattr(model, "pos_encoding")
    assert hasattr(model, "decoder")
    assert hasattr(model, "lm_head")
    assert hasattr(model, "ln_f")

    # Check parameter count
    num_params = model.get_num_params()
    assert num_params > 0


def test_latent_decoder_forward():
    """Test LatentDecoder forward pass."""
    config = LatentDecoderConfig(
        latent_dim=512,
        vocab_size=1000,
        decoder_dim=256,
        depth=2,
        max_seq_len=128,
    )

    model = LatentDecoder(config)

    batch_size = 4
    seq_len = 20
    latent = torch.randn(batch_size, config.latent_dim)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward without labels (inference)
    outputs = model(latent, input_ids)

    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert "loss" not in outputs  # No loss without labels

    # Forward with labels (training)
    labels = input_ids.clone()
    labels[:, :5] = -100  # Mask first 5 tokens

    outputs = model(latent, input_ids, labels)

    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["loss"].item() >= 0  # Loss should be non-negative


def test_latent_decoder_generate():
    """Test LatentDecoder generate method."""
    config = LatentDecoderConfig(
        latent_dim=512,
        vocab_size=1000,
        decoder_dim=256,
        depth=2,
    )

    model = LatentDecoder(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    latent = torch.randn(1, config.latent_dim)

    # Test generate
    with torch.no_grad():
        result = model.generate(
            latent,
            tokenizer,
            max_new_tokens=10,
            temperature=0.8,
            top_p=0.9,
        )

    assert "generated_ids" in result
    assert "text" in result
    assert result["generated_ids"].shape[0] == 1
    assert result["generated_ids"].shape[1] <= 10 + 1  # +1 for BOS token
    assert isinstance(result["text"], str)


def test_create_latent_decoder():
    """Test create_latent_decoder factory function."""
    model = create_latent_decoder(
        latent_dim=512,
        vocab_size=1000,
        decoder_dim=256,
        depth=2,
    )

    assert isinstance(model, LatentDecoder)
    assert model.config.latent_dim == 512
    assert model.config.vocab_size == 1000


def test_weight_tying():
    """Test weight tying between embeddings and LM head."""
    config = LatentDecoderConfig(
        latent_dim=512,
        vocab_size=1000,
        decoder_dim=256,
        depth=2,
        tie_embeddings=True,
    )

    model = LatentDecoder(config)

    # Check that weights are shared
    assert model.token_embed.weight is model.lm_head.weight

    # Check that gradients flow correctly
    batch_size = 2
    seq_len = 10
    latent = torch.randn(batch_size, config.latent_dim)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(latent, input_ids, labels)
    loss = outputs["loss"]
    loss.backward()

    # Both should have gradients
    assert model.token_embed.weight.grad is not None
    assert model.lm_head.weight.grad is not None


# ============================================================================
# Test Trainer
# ============================================================================

def test_trainer_initialization():
    """Test LatentDecoderTrainer initialization."""
    from torch.utils.data import DataLoader, TensorDataset

    config = LatentDecoderConfig(
        latent_dim=512,
        vocab_size=1000,
        decoder_dim=256,
        depth=2,
    )

    model = LatentDecoder(config)

    # Create dummy dataset
    dummy_latents = torch.randn(50, config.latent_dim)
    dummy_input_ids = torch.randint(0, config.vocab_size, (50, 20))
    dummy_labels = dummy_input_ids.clone()

    dataset = TensorDataset(dummy_latents, dummy_input_ids, dummy_labels)
    loader = DataLoader(dataset, batch_size=8)

    # Create trainer
    trainer = LatentDecoderTrainer(
        model=model,
        train_loader=loader,
        lr=1e-4,
        device="cpu",
        use_amp=False,
        wandb_enabled=False,
    )

    assert trainer.model is model
    assert trainer.device == "cpu"
    assert trainer.use_amp is False
    assert trainer.optimizer is not None
    assert trainer.scaler is None  # No scaler when AMP disabled


def test_trainer_single_epoch():
    """Test training for one epoch."""
    from torch.utils.data import DataLoader, TensorDataset

    config = LatentDecoderConfig(
        latent_dim=512,
        vocab_size=1000,
        decoder_dim=256,
        depth=2,
    )

    model = LatentDecoder(config)

    # Create dummy dataset
    dummy_latents = torch.randn(50, config.latent_dim)
    dummy_input_ids = torch.randint(0, config.vocab_size, (50, 20))
    dummy_labels = dummy_input_ids.clone()

    dataset = TensorDataset(dummy_latents, dummy_input_ids, dummy_labels)
    loader = DataLoader(dataset, batch_size=8)

    trainer = LatentDecoderTrainer(
        model=model,
        train_loader=loader,
        lr=1e-4,
        device="cpu",
        use_amp=False,
        wandb_enabled=False,
    )

    # Train one epoch
    metrics = trainer.train_epoch()

    assert "loss" in metrics
    assert "perplexity" in metrics
    assert metrics["loss"] >= 0
    assert metrics["perplexity"] >= 1.0  # Perplexity = exp(loss) >= 1


def test_trainer_validation():
    """Test validation loop."""
    from torch.utils.data import DataLoader, TensorDataset

    config = LatentDecoderConfig(
        latent_dim=512,
        vocab_size=1000,
        decoder_dim=256,
        depth=2,
    )

    model = LatentDecoder(config)

    # Create dummy datasets
    train_data = TensorDataset(
        torch.randn(50, config.latent_dim),
        torch.randint(0, config.vocab_size, (50, 20)),
        torch.randint(0, config.vocab_size, (50, 20)),
    )
    val_data = TensorDataset(
        torch.randn(20, config.latent_dim),
        torch.randint(0, config.vocab_size, (20, 20)),
        torch.randint(0, config.vocab_size, (20, 20)),
    )

    train_loader = DataLoader(train_data, batch_size=8)
    val_loader = DataLoader(val_data, batch_size=8)

    trainer = LatentDecoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-4,
        device="cpu",
        use_amp=False,
        wandb_enabled=False,
    )

    # Validate
    val_metrics = trainer.validate()

    assert "loss" in val_metrics
    assert "perplexity" in val_metrics
    assert val_metrics["loss"] >= 0


def test_trainer_checkpointing():
    """Test checkpoint save/load."""
    import tempfile
    from pathlib import Path
    from torch.utils.data import DataLoader, TensorDataset

    config = LatentDecoderConfig(
        latent_dim=512,
        vocab_size=1000,
        decoder_dim=256,
        depth=2,
    )

    model = LatentDecoder(config)

    dummy_data = TensorDataset(
        torch.randn(50, config.latent_dim),
        torch.randint(0, config.vocab_size, (50, 20)),
        torch.randint(0, config.vocab_size, (50, 20)),
    )
    loader = DataLoader(dummy_data, batch_size=8)

    trainer = LatentDecoderTrainer(
        model=model,
        train_loader=loader,
        lr=1e-4,
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
        new_model = LatentDecoder(config)
        new_trainer = LatentDecoderTrainer(
            model=new_model,
            train_loader=loader,
            lr=1e-4,
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
    from torch.utils.data import DataLoader, TensorDataset

    config = LatentDecoderConfig(
        latent_dim=512,
        vocab_size=1000,
        decoder_dim=256,
        depth=2,
    )

    model = LatentDecoder(config)

    dummy_data = TensorDataset(
        torch.randn(50, config.latent_dim),
        torch.randint(0, config.vocab_size, (50, 20)),
        torch.randint(0, config.vocab_size, (50, 20)),
    )
    loader = DataLoader(dummy_data, batch_size=8)

    trainer = LatentDecoderTrainer(
        model=model,
        train_loader=loader,
        lr=1e-4,
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
        assert "train_perplexity" in history
        assert len(history["train_loss"]) == 2
        assert len(history["train_perplexity"]) == 2

        # Check checkpoints were saved
        checkpoints = list(Path(tmpdir).glob("checkpoint-epoch-*.pth"))
        assert len(checkpoints) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
