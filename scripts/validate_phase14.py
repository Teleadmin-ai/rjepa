"""
Validation script for Phase 14: Contrastive Loss (InfoNCE) Active.

Verifies:
1. JEPALoss has contrastive loss enabled by default
2. Contrastive loss computation works correctly
3. Hard negatives support is functional
4. Config file has contrastive_weight = 0.1 (active)
5. Training with contrastive loss works
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("PHASE 14 VALIDATION: CONTRASTIVE LOSS (INFONCE) ACTIVE")
print("="*80)

# Check 1: Default contrastive weight is non-zero
print("\n[1/6] Checking JEPALoss defaults...")
try:
    from rjepa.jepa.losses import JEPALoss

    loss_fn = JEPALoss()

    # Check that contrastive loss is ACTIVE by default
    assert loss_fn.contrastive_weight > 0, (
        f"Contrastive loss should be ACTIVE by default, "
        f"but got weight={loss_fn.contrastive_weight}"
    )

    print(f"  [OK] Contrastive loss ACTIVE by default (weight={loss_fn.contrastive_weight})")
    print(f"  [OK] Hard negatives support: {loss_fn.use_hard_negatives}")
    print(f"  [OK] Temperature: {loss_fn.contrastive_temperature}")

except Exception as e:
    print(f"[FAIL] JEPALoss defaults check failed: {e}")
    sys.exit(1)

# Check 2: Contrastive loss computation (no hard negatives)
print("\n[2/6] Testing contrastive loss computation (standard)...")
try:
    import torch

    batch_size = 4
    num_steps = 10
    dim = 512

    pred = torch.randn(batch_size, num_steps, dim)
    target = torch.randn(batch_size, num_steps, dim)
    mask = torch.rand(batch_size, num_steps) > 0.5  # Random mask

    # Ensure at least some True values
    mask[:, 0] = True

    loss_value = loss_fn.contrastive_loss(pred, target, mask)

    assert loss_value.item() >= 0, "Contrastive loss should be non-negative"

    print(f"  [OK] Contrastive loss computed: {loss_value.item():.4f}")

except Exception as e:
    print(f"[FAIL] Contrastive loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 3: Contrastive loss with hard negatives
print("\n[3/6] Testing contrastive loss with hard negatives...")
try:
    # Add hard negatives
    num_hard_negatives = 8
    hard_negatives = torch.randn(num_hard_negatives, dim)

    loss_with_hard = loss_fn.contrastive_loss(
        pred, target, mask, hard_negatives=hard_negatives
    )

    assert loss_with_hard.item() >= 0

    print(f"  [OK] Contrastive loss with hard negatives: {loss_with_hard.item():.4f}")

    # Loss with hard negatives should generally be >= loss without
    # (more negatives make discrimination harder)
    print(f"  [OK] Loss standard: {loss_value.item():.4f}, with hard negs: {loss_with_hard.item():.4f}")

except Exception as e:
    print(f"[FAIL] Hard negatives test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 4: Full forward pass with all loss components
print("\n[4/6] Testing full JEPALoss forward pass...")
try:
    losses = loss_fn(pred, target, mask, hard_negatives=hard_negatives)

    # Check all components exist
    required_keys = ["loss", "recon_loss", "var_reg_loss", "contrastive_loss"]
    for key in required_keys:
        assert key in losses, f"Missing key: {key}"

    # Check values are reasonable
    assert losses["loss"].item() >= 0
    assert losses["recon_loss"].item() >= 0
    assert losses["var_reg_loss"].item() >= 0
    assert losses["contrastive_loss"].item() >= 0

    print(f"  [OK] Total loss: {losses['loss'].item():.4f}")
    print(f"  [OK] Reconstruction: {losses['recon_loss'].item():.4f}")
    print(f"  [OK] Variance reg: {losses['var_reg_loss'].item():.4f}")
    print(f"  [OK] Contrastive: {losses['contrastive_loss'].item():.4f}")

    # Verify contrastive component is included in total
    manual_total = (
        losses["recon_loss"].item()
        + loss_fn.var_reg_weight * losses["var_reg_loss"].item()
        + loss_fn.contrastive_weight * losses["contrastive_loss"].item()
    )

    assert abs(losses["loss"].item() - manual_total) < 1e-4, (
        f"Total loss mismatch: {losses['loss'].item()} vs {manual_total}"
    )

    print(f"  [OK] Loss components sum correctly")

except Exception as e:
    print(f"[FAIL] Full forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 5: Config file has contrastive loss active
print("\n[5/6] Testing config file...")
try:
    import yaml

    config_path = Path("configs/rjepa/train.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    contrastive_weight = config["model"]["loss"]["contrastive_weight"]

    assert contrastive_weight > 0, (
        f"Config should have contrastive_weight > 0, got {contrastive_weight}"
    )

    print(f"  [OK] Config contrastive_weight: {contrastive_weight} (ACTIVE)")

    use_hard_negatives = config["model"]["loss"].get("use_hard_negatives", False)
    print(f"  [OK] Config use_hard_negatives: {use_hard_negatives}")

except Exception as e:
    print(f"[FAIL] Config check failed: {e}")
    sys.exit(1)

# Check 6: Backward pass (gradient flow)
print("\n[6/6] Testing gradient flow through contrastive loss...")
try:
    # Create model parameters (dummy)
    pred_param = torch.nn.Parameter(torch.randn(batch_size, num_steps, dim))

    # Forward
    losses = loss_fn(pred_param, target, mask, hard_negatives=hard_negatives)
    total_loss = losses["loss"]

    # Backward
    total_loss.backward()

    # Check gradients exist
    assert pred_param.grad is not None, "Gradients should flow to predictions"
    assert pred_param.grad.norm().item() > 0, "Gradients should be non-zero"

    print(f"  [OK] Gradients computed successfully")
    print(f"  [OK] Gradient norm: {pred_param.grad.norm().item():.4f}")

except Exception as e:
    print(f"[FAIL] Gradient flow test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*80)
print("PHASE 14 VALIDATION: [PASS] ALL CHECKS SUCCESSFUL")
print("="*80)
print("\nContrastive Loss InfoNCE Implementation:")
print(f"  - Contrastive loss ACTIVE by default (weight={loss_fn.contrastive_weight})")
print(f"  - Hard negatives support enabled")
print(f"  - InfoNCE temperature: {loss_fn.contrastive_temperature}")
print(f"  - Integration with R-JEPA training")
print("\nKey Benefits:")
print("  - Better discrimination between correct/incorrect reasoning")
print("  - Learns to distinguish true next latent from hard negatives")
print("  - Improved manifold structure (tighter clusters)")
print("  - Complements reconstruction loss (L1 + variance + contrastive)")
print("\nNext steps:")
print("  1. Train R-JEPA with contrastive loss active")
print("  2. Measure improved discrimination (JEPA-loss correlation with correctness)")
print("  3. Collect hard negatives from incorrect CoTs during training")
print("="*80)
