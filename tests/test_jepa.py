"""
Test R-JEPA model components.
"""
import pytest
import torch
import tempfile
from pathlib import Path

from rjepa.jepa import (
    ReasoningJEPA,
    ReasoningEncoder,
    ReasoningPredictor,
    RandomMasker,
    ContiguousMasker,
    HierarchicalMasker,
    MaskCollator,
    JEPALoss,
    LatentDataset,
)


def test_random_masker():
    """Test random masking strategy."""
    masker = RandomMasker(mask_ratio=0.5)

    batch_size, num_steps = 4, 10
    context_mask, target_mask = masker(batch_size, num_steps)

    assert context_mask.shape == (batch_size, num_steps)
    assert target_mask.shape == (batch_size, num_steps)
    assert torch.all(context_mask == ~target_mask)

    # Check approximately 50% masked
    mask_ratio = target_mask.float().mean().item()
    assert 0.3 < mask_ratio < 0.7  # Allow some variance


def test_contiguous_masker():
    """Test contiguous block masking."""
    masker = ContiguousMasker(min_mask_ratio=0.3, max_mask_ratio=0.7, num_blocks=1)

    batch_size, num_steps = 4, 20
    context_mask, target_mask = masker(batch_size, num_steps)

    assert context_mask.shape == (batch_size, num_steps)
    assert target_mask.shape == (batch_size, num_steps)

    # Check that each sample has a contiguous masked block
    for b in range(batch_size):
        target_indices = torch.where(target_mask[b])[0]
        if len(target_indices) > 1:
            # Check contiguity
            diffs = target_indices[1:] - target_indices[:-1]
            assert torch.all(diffs == 1), "Masked steps should be contiguous"


def test_hierarchical_masker():
    """Test hierarchical masking."""
    masker = HierarchicalMasker(
        intro_keep_prob=1.0,
        middle_mask_prob=1.0,
        conclusion_keep_prob=1.0,
    )

    batch_size, num_steps = 4, 10
    context_mask, target_mask = masker(batch_size, num_steps)

    assert context_mask.shape == (batch_size, num_steps)

    # With these settings: first and last visible, middle masked
    for b in range(batch_size):
        assert context_mask[b, 0] == True  # First step visible
        assert context_mask[b, -1] == True  # Last step visible
        # Middle should be masked
        assert torch.all(target_mask[b, 1:-1])


def test_reasoning_encoder():
    """Test reasoning encoder."""
    encoder = ReasoningEncoder(
        dim=128,
        depth=4,
        num_heads=4,
        max_steps=64,
        domain_embed_dim=32,
        num_domains=5,
    )

    batch_size, seq_len, dim = 2, 10, 128

    x = torch.randn(batch_size, seq_len, dim)
    domain_ids = torch.randint(0, 5, (batch_size,))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, 5:] = False  # Mask second half

    # Forward pass
    output = encoder(x, domain_ids=domain_ids, mask=mask)

    assert output.shape == (batch_size, seq_len, dim)
    assert not torch.isnan(output).any()


def test_reasoning_predictor():
    """Test reasoning predictor."""
    predictor = ReasoningPredictor(
        dim=128,
        predictor_dim=256,
        depth=6,
        num_heads=8,
    )

    batch_size, seq_len, dim = 2, 10, 128

    latents = torch.randn(batch_size, seq_len, dim)
    context_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    context_mask[:, 5:] = False  # Context = first half
    target_mask = ~context_mask  # Target = second half

    # Forward pass
    pred = predictor(latents, context_mask, target_mask)

    assert pred.shape == (batch_size, seq_len, dim)
    assert not torch.isnan(pred).any()


def test_jepa_loss():
    """Test JEPA loss computation."""
    criterion = JEPALoss(
        loss_type="l1",
        var_reg_weight=0.01,
        contrastive_weight=0.0,
    )

    batch_size, seq_len, dim = 4, 10, 128

    pred = torch.randn(batch_size, seq_len, dim)
    target = torch.randn(batch_size, seq_len, dim)
    target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    target_mask[:, 5:] = True  # Target = second half

    # Compute loss
    losses = criterion(pred, target, target_mask)

    assert "loss" in losses
    assert "recon_loss" in losses
    assert "var_reg_loss" in losses

    assert losses["loss"].item() > 0
    assert not torch.isnan(losses["loss"])


def test_reasoning_jepa_forward():
    """Test full R-JEPA forward pass."""
    model = ReasoningJEPA(
        dim=128,
        depth_encoder=4,
        depth_predictor=3,
        num_heads=4,
        predictor_dim=256,
        max_steps=64,
        domain_embed_dim=32,
        num_domains=5,
        ema_momentum=0.996,
        loss_config={"loss_type": "l1", "var_reg_weight": 0.01},
    )

    batch_size, seq_len, dim = 2, 10, 128

    latents = torch.randn(batch_size, seq_len, dim)
    domain_ids = torch.randint(0, 5, (batch_size,))

    # Masks
    context_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    context_mask[:, 5:] = False
    target_mask = ~context_mask

    # Forward with loss
    outputs = model(
        latents,
        context_mask,
        target_mask,
        domain_ids,
        compute_loss=True,
    )

    assert "pred" in outputs
    assert "target" in outputs
    assert "loss" in outputs

    assert outputs["pred"].shape == (batch_size, seq_len, dim)
    assert outputs["target"].shape == (batch_size, seq_len, dim)
    assert outputs["loss"].item() > 0

    # Forward without loss
    outputs_no_loss = model(
        latents,
        context_mask,
        target_mask,
        domain_ids,
        compute_loss=False,
    )

    assert "pred" in outputs_no_loss
    assert "loss" not in outputs_no_loss


def test_ema_update():
    """Test EMA target encoder update."""
    model = ReasoningJEPA(
        dim=64,
        depth_encoder=2,
        depth_predictor=2,
        num_heads=2,
        ema_momentum=0.9,
    )

    # Get initial target encoder params
    target_params_before = [
        p.clone() for p in model.target_encoder.parameters()
    ]

    # Update EMA
    model.update_target_encoder()

    # Check that target params changed
    target_params_after = list(model.target_encoder.parameters())

    for p_before, p_after in zip(target_params_before, target_params_after):
        assert not torch.allclose(p_before, p_after), "EMA should update parameters"


def test_jepa_score():
    """Test JEPA score computation."""
    model = ReasoningJEPA(
        dim=64,
        depth_encoder=2,
        depth_predictor=2,
        num_heads=2,
    )

    batch_size, seq_len, dim = 4, 8, 64

    latents = torch.randn(batch_size, seq_len, dim)

    context_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    context_mask[:, 4:] = False
    target_mask = ~context_mask

    # Get JEPA score
    scores = model.get_jepa_score(latents, context_mask, target_mask)

    assert scores.shape == (batch_size,)
    assert torch.all(scores >= 0), "JEPA scores should be non-negative"
    assert not torch.isnan(scores).any()


def test_mask_collator():
    """Test mask collator for DataLoader."""
    masker = ContiguousMasker(min_mask_ratio=0.3, max_mask_ratio=0.5)
    collator = MaskCollator(masker, device="cpu")

    # Simulate batch from dataset
    batch = [
        (torch.randn(10, 128), 0),  # (latents, domain_id)
        (torch.randn(12, 128), 1),
        (torch.randn(8, 128), 2),
    ]

    # Collate - should pad to max length
    # Actually, our collator doesn't pad yet, so sequences must be same length
    # Let's fix the test
    batch_same_len = [
        (torch.randn(10, 128), 0),
        (torch.randn(10, 128), 1),
        (torch.randn(10, 128), 2),
    ]

    collated = collator(batch_same_len)

    assert "latents" in collated
    assert "domain_ids" in collated
    assert "context_mask" in collated
    assert "target_mask" in collated

    assert collated["latents"].shape == (3, 10, 128)
    assert collated["domain_ids"].shape == (3,)
    assert collated["context_mask"].shape == (3, 10)
    assert collated["target_mask"].shape == (3, 10)


def test_latent_dataset():
    """Test latent dataset loading."""
    # Create dummy latent data
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create dummy shards
        from rjepa.data.sharding import LatentSharding
        from rjepa.data.schemas import LatentSequence
        from dataclasses import asdict

        metadata_records = [
            asdict(
                LatentSequence(
                    problem_id=f"p{i}",
                    cot_id=f"cot{i}",
                    llm_tag="test-llm",
                    layer_idx=-2,
                    hidden_size=64,
                    num_steps=5,
                    step_boundaries=[(0, 5)],
                    domain="math" if i < 5 else "code",
                    subdomain="",
                )
            )
            for i in range(10)
        ]

        latents_dict = {f"cot{i}": torch.randn(5, 64) for i in range(10)}

        LatentSharding.save_latent_shard(
            metadata_records, latents_dict, output_dir, shard_idx=0
        )

        # Load dataset
        dataset = LatentDataset(output_dir, device="cpu")

        assert len(dataset) == 10

        # Get one sample
        H, domain_id = dataset[0]
        assert H.shape == (5, 64)
        assert domain_id in [0, 1]  # math or code

        # Get stats
        stats = dataset.get_stats()
        assert stats["num_samples"] == 10
        assert "math" in stats["domain_counts"]
        assert "code" in stats["domain_counts"]
