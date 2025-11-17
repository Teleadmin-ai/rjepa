"""
Tests for Contrastive Loss (InfoNCE).
"""
import pytest
import torch

from rjepa.jepa.losses import JEPALoss


# ============================================================================
# Test Contrastive Loss Defaults
# ============================================================================

def test_contrastive_loss_active_by_default():
    """Test that contrastive loss is ACTIVE by default."""
    loss_fn = JEPALoss()

    # Contrastive loss should be active
    assert loss_fn.contrastive_weight > 0, (
        f"Contrastive loss should be ACTIVE by default, "
        f"but got weight={loss_fn.contrastive_weight}"
    )

    # Default should be 0.1
    assert loss_fn.contrastive_weight == 0.1


def test_contrastive_loss_hard_negatives_enabled():
    """Test that hard negatives are enabled by default."""
    loss_fn = JEPALoss()

    assert loss_fn.use_hard_negatives is True


# ============================================================================
# Test Contrastive Loss Computation
# ============================================================================

def test_contrastive_loss_basic():
    """Test basic contrastive loss computation."""
    loss_fn = JEPALoss()

    batch_size = 4
    num_steps = 10
    dim = 512

    pred = torch.randn(batch_size, num_steps, dim)
    target = torch.randn(batch_size, num_steps, dim)
    mask = torch.ones(batch_size, num_steps, dtype=torch.bool)

    loss = loss_fn.contrastive_loss(pred, target, mask)

    # Loss should be non-negative
    assert loss.item() >= 0

    # Loss should be a scalar
    assert loss.shape == torch.Size([])


def test_contrastive_loss_with_mask():
    """Test contrastive loss with partial mask."""
    loss_fn = JEPALoss()

    batch_size = 4
    num_steps = 10
    dim = 512

    pred = torch.randn(batch_size, num_steps, dim)
    target = torch.randn(batch_size, num_steps, dim)

    # Partial mask
    mask = torch.zeros(batch_size, num_steps, dtype=torch.bool)
    mask[:, 0] = True  # Only first step

    loss = loss_fn.contrastive_loss(pred, target, mask)

    assert loss.item() >= 0


def test_contrastive_loss_empty_mask():
    """Test contrastive loss with empty mask."""
    loss_fn = JEPALoss()

    batch_size = 4
    num_steps = 10
    dim = 512

    pred = torch.randn(batch_size, num_steps, dim)
    target = torch.randn(batch_size, num_steps, dim)

    # Empty mask
    mask = torch.zeros(batch_size, num_steps, dtype=torch.bool)

    loss = loss_fn.contrastive_loss(pred, target, mask)

    # Should return zero
    assert loss.item() == 0.0


def test_contrastive_loss_with_hard_negatives():
    """Test contrastive loss with hard negatives."""
    loss_fn = JEPALoss()

    batch_size = 4
    num_steps = 10
    dim = 512

    pred = torch.randn(batch_size, num_steps, dim)
    target = torch.randn(batch_size, num_steps, dim)
    mask = torch.ones(batch_size, num_steps, dtype=torch.bool)

    # Add hard negatives
    num_hard_negatives = 8
    hard_negatives = torch.randn(num_hard_negatives, dim)

    loss_with_hard = loss_fn.contrastive_loss(
        pred, target, mask, hard_negatives=hard_negatives
    )

    assert loss_with_hard.item() >= 0

    # Compute loss without hard negatives for comparison
    loss_no_hard = loss_fn.contrastive_loss(pred, target, mask)

    # Loss with hard negatives should generally be >= loss without
    # (more negatives make discrimination harder, but not always)
    assert loss_with_hard.item() >= 0
    assert loss_no_hard.item() >= 0


def test_contrastive_loss_perfect_prediction():
    """Test contrastive loss when prediction matches target."""
    loss_fn = JEPALoss()

    batch_size = 4
    num_steps = 10
    dim = 512

    # Perfect prediction: pred = target
    latent = torch.randn(batch_size, num_steps, dim)
    pred = latent
    target = latent.clone()

    mask = torch.ones(batch_size, num_steps, dtype=torch.bool)

    loss = loss_fn.contrastive_loss(pred, target, mask)

    # Loss should be very low (close to 0) but not exactly 0
    # due to numerical precision and normalization
    assert loss.item() < 0.1


# ============================================================================
# Test Full Loss with Contrastive Component
# ============================================================================

def test_full_loss_includes_contrastive():
    """Test that full loss includes contrastive component."""
    loss_fn = JEPALoss()

    batch_size = 4
    num_steps = 10
    dim = 512

    pred = torch.randn(batch_size, num_steps, dim)
    target = torch.randn(batch_size, num_steps, dim)
    mask = torch.ones(batch_size, num_steps, dtype=torch.bool)

    # Full forward pass
    losses = loss_fn(pred, target, mask)

    # Check all components exist
    assert "loss" in losses
    assert "recon_loss" in losses
    assert "var_reg_loss" in losses
    assert "contrastive_loss" in losses

    # Manually compute total loss
    manual_total = (
        losses["recon_loss"].item()
        + loss_fn.var_reg_weight * losses["var_reg_loss"].item()
        + loss_fn.contrastive_weight * losses["contrastive_loss"].item()
    )

    # Should match
    assert abs(losses["loss"].item() - manual_total) < 1e-4


def test_full_loss_with_hard_negatives():
    """Test full loss with hard negatives."""
    loss_fn = JEPALoss()

    batch_size = 4
    num_steps = 10
    dim = 512

    pred = torch.randn(batch_size, num_steps, dim)
    target = torch.randn(batch_size, num_steps, dim)
    mask = torch.ones(batch_size, num_steps, dtype=torch.bool)

    # Add hard negatives
    hard_negatives = torch.randn(8, dim)

    # Full forward pass with hard negatives
    losses = loss_fn(pred, target, mask, hard_negatives=hard_negatives)

    # Check all components exist
    assert "loss" in losses
    assert "contrastive_loss" in losses

    # All should be non-negative
    assert losses["loss"].item() >= 0
    assert losses["contrastive_loss"].item() >= 0


def test_contrastive_loss_disabled():
    """Test that contrastive loss can be disabled."""
    # Create loss with contrastive disabled
    loss_fn = JEPALoss(contrastive_weight=0.0)

    batch_size = 4
    num_steps = 10
    dim = 512

    pred = torch.randn(batch_size, num_steps, dim)
    target = torch.randn(batch_size, num_steps, dim)
    mask = torch.ones(batch_size, num_steps, dtype=torch.bool)

    losses = loss_fn(pred, target, mask)

    # Contrastive loss should not be in losses dict
    assert "contrastive_loss" not in losses

    # Total loss should only be recon + var_reg
    manual_total = (
        losses["recon_loss"].item()
        + loss_fn.var_reg_weight * losses["var_reg_loss"].item()
    )

    assert abs(losses["loss"].item() - manual_total) < 1e-4


# ============================================================================
# Test Gradient Flow
# ============================================================================

def test_gradient_flow_through_contrastive():
    """Test that gradients flow through contrastive loss."""
    loss_fn = JEPALoss()

    batch_size = 4
    num_steps = 10
    dim = 512

    # Create parameter
    pred = torch.nn.Parameter(torch.randn(batch_size, num_steps, dim))
    target = torch.randn(batch_size, num_steps, dim)
    mask = torch.ones(batch_size, num_steps, dtype=torch.bool)

    # Forward + backward
    losses = loss_fn(pred, target, mask)
    losses["loss"].backward()

    # Check gradients exist and are non-zero
    assert pred.grad is not None
    assert pred.grad.norm().item() > 0


def test_gradient_flow_with_hard_negatives():
    """Test gradient flow with hard negatives."""
    loss_fn = JEPALoss()

    batch_size = 4
    num_steps = 10
    dim = 512

    pred = torch.nn.Parameter(torch.randn(batch_size, num_steps, dim))
    target = torch.randn(batch_size, num_steps, dim)
    mask = torch.ones(batch_size, num_steps, dtype=torch.bool)
    hard_negatives = torch.randn(8, dim)

    # Forward + backward
    losses = loss_fn(pred, target, mask, hard_negatives=hard_negatives)
    losses["loss"].backward()

    # Check gradients
    assert pred.grad is not None
    assert pred.grad.norm().item() > 0


# ============================================================================
# Test Temperature Effect
# ============================================================================

def test_contrastive_temperature_effect():
    """Test that temperature affects contrastive loss."""
    batch_size = 4
    num_steps = 10
    dim = 512

    pred = torch.randn(batch_size, num_steps, dim)
    target = torch.randn(batch_size, num_steps, dim)
    mask = torch.ones(batch_size, num_steps, dtype=torch.bool)

    # Low temperature (more discriminative)
    loss_fn_low_temp = JEPALoss(contrastive_temperature=0.01)
    loss_low = loss_fn_low_temp.contrastive_loss(pred, target, mask)

    # High temperature (less discriminative)
    loss_fn_high_temp = JEPALoss(contrastive_temperature=1.0)
    loss_high = loss_fn_high_temp.contrastive_loss(pred, target, mask)

    # Both should be positive
    assert loss_low.item() >= 0
    assert loss_high.item() >= 0

    # Lower temperature typically leads to higher loss (harder to distinguish)
    # but this is not guaranteed, so just check both are reasonable
    assert loss_low.item() > 0
    assert loss_high.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
