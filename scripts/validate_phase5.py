#!/usr/bin/env python3
"""
Validate Phase 5: R-JEPA Model
"""
import sys
import torch
from pathlib import Path


def validate_phase5():
    """Validate that Phase 5 is complete."""
    print("[*] Validating Phase 5: R-JEPA Model...")
    print()

    # Check files exist
    required_files = [
        "rjepa/jepa/__init__.py",
        "rjepa/jepa/model.py",
        "rjepa/jepa/encoder.py",
        "rjepa/jepa/predictor.py",
        "rjepa/jepa/maskers.py",
        "rjepa/jepa/losses.py",
        "rjepa/jepa/dataset.py",
        "tests/test_jepa.py",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[FAIL] {file_path} (MISSING)")
            all_exist = False

    print()

    if not all_exist:
        print("[FAIL] Phase 5 validation FAILED: Some files are missing")
        return False

    # Try importing
    try:
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
        print("[OK] All R-JEPA components import successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import R-JEPA components: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test instantiation
    try:
        print("[*] Testing ContiguousMasker...")
        masker = ContiguousMasker(min_mask_ratio=0.3, max_mask_ratio=0.7)
        context_mask, target_mask = masker(batch_size=2, num_steps=10)
        assert context_mask.shape == (2, 10)
        assert target_mask.shape == (2, 10)
        print("[OK] ContiguousMasker works")

    except Exception as e:
        print(f"[FAIL] ContiguousMasker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        print("[*] Testing ReasoningEncoder...")
        encoder = ReasoningEncoder(
            dim=128,
            depth=4,
            num_heads=4,
            max_steps=64,
            domain_embed_dim=32,
            num_domains=5,
        )

        x = torch.randn(2, 10, 128)
        domain_ids = torch.randint(0, 5, (2,))
        mask = torch.ones(2, 10, dtype=torch.bool)

        output = encoder(x, domain_ids=domain_ids, mask=mask)
        assert output.shape == (2, 10, 128)

        print("[OK] ReasoningEncoder works")

    except Exception as e:
        print(f"[FAIL] ReasoningEncoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        print("[*] Testing ReasoningPredictor...")
        predictor = ReasoningPredictor(
            dim=128,
            predictor_dim=256,
            depth=6,
            num_heads=8,
        )

        latents = torch.randn(2, 10, 128)
        context_mask = torch.ones(2, 10, dtype=torch.bool)
        context_mask[:, 5:] = False
        target_mask = ~context_mask

        pred = predictor(latents, context_mask, target_mask)
        assert pred.shape == (2, 10, 128)

        print("[OK] ReasoningPredictor works")

    except Exception as e:
        print(f"[FAIL] ReasoningPredictor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        print("[*] Testing JEPALoss...")
        criterion = JEPALoss(
            loss_type="l1",
            var_reg_weight=0.01,
        )

        pred = torch.randn(4, 10, 128)
        target = torch.randn(4, 10, 128)
        target_mask = torch.zeros(4, 10, dtype=torch.bool)
        target_mask[:, 5:] = True

        losses = criterion(pred, target, target_mask)

        assert "loss" in losses
        assert "recon_loss" in losses
        assert "var_reg_loss" in losses
        assert losses["loss"].item() > 0

        print("[OK] JEPALoss works")

    except Exception as e:
        print(f"[FAIL] JEPALoss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        print("[*] Testing ReasoningJEPA (full model)...")
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

        latents = torch.randn(2, 10, 128)
        domain_ids = torch.randint(0, 5, (2,))
        context_mask = torch.ones(2, 10, dtype=torch.bool)
        context_mask[:, 5:] = False
        target_mask = ~context_mask

        # Forward pass
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
        assert outputs["loss"].item() > 0

        # EMA update
        model.update_target_encoder()

        # JEPA score
        scores = model.get_jepa_score(latents, context_mask, target_mask, domain_ids)
        assert scores.shape == (2,)

        print("[OK] ReasoningJEPA works (forward + EMA + scoring)")

    except Exception as e:
        print(f"[FAIL] ReasoningJEPA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 80)
    print("SUCCESS: PHASE 5 VALIDATION COMPLETE!")
    print("=" * 80)
    print()
    print("[OK] All required files exist")
    print("[OK] All imports work")
    print("[OK] ContiguousMasker works")
    print("[OK] ReasoningEncoder works")
    print("[OK] ReasoningPredictor works")
    print("[OK] JEPALoss works")
    print("[OK] ReasoningJEPA full model works")
    print()
    print("Statistics:")
    print("   - Maskers: 3 (Random, Contiguous, Hierarchical)")
    print("   - Encoder: Transformer-based (depth configurable)")
    print("   - Predictor: Transformer-based with mask tokens")
    print("   - Loss: L1 + variance reg + optional contrastive")
    print("   - Dataset: LatentDataset (loads from sharded parquet/safetensors)")
    print("   - Model: ReasoningJEPA with EMA target encoder")
    print()
    print("Key Features:")
    print("   [OK] World model architecture (encoder + predictor + EMA)")
    print("   [OK] Contiguous masking (recommended for reasoning)")
    print("   [OK] Domain embeddings (multi-domain support)")
    print("   [OK] JEPA score for inference (re-ranking, nudging)")
    print("   [OK] Variance regularization (prevent collapse)")
    print()
    print("READY FOR PHASE 6: Training Pipeline (trainer, checkpointing, W&B)")
    print()

    return True


if __name__ == "__main__":
    success = validate_phase5()
    sys.exit(0 if success else 1)
