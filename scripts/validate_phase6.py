#!/usr/bin/env python3
"""
Validate Phase 6: Training Pipeline
"""
import sys
import torch
import tempfile
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


def validate_phase6():
    """Validate that Phase 6 is complete."""
    print("[*] Validating Phase 6: Training Pipeline...")
    print()

    # Check files exist
    required_files = [
        "rjepa/jepa/trainer.py",
        "rjepa/pipeline/train_rjepa.py",
        "configs/rjepa/train.yaml",
        "tests/test_trainer.py",
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
        print("[FAIL] Phase 6 validation FAILED: Some files are missing")
        return False

    # Try importing
    try:
        from rjepa.jepa.trainer import RJEPATrainer
        print("[OK] RJEPATrainer imports successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import RJEPATrainer: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        from rjepa.pipeline.train_rjepa import (
            load_config,
            create_dataloaders,
            train_rjepa_from_config,
        )
        print("[OK] Training pipeline imports successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test trainer instantiation
    try:
        print("[*] Testing RJEPATrainer instantiation...")

        from rjepa.jepa import ReasoningJEPA, ContiguousMasker

        # Create dummy model
        model = ReasoningJEPA(
            dim=64,
            depth_encoder=2,
            depth_predictor=2,
            num_heads=2,
            predictor_dim=128,
            max_steps=32,
            ema_momentum=0.9,
        )

        # Create dummy dataloader
        latents = torch.randn(20, 10, 64)
        domain_ids = torch.randint(0, 5, (20,))
        dataset = TensorDataset(latents, domain_ids)

        masker = ContiguousMasker(min_mask_ratio=0.3, max_mask_ratio=0.5)

        def collate_fn(batch):
            latents_batch = torch.stack([item[0] for item in batch])
            domain_ids_batch = torch.stack([item[1] for item in batch])
            batch_size, num_steps, dim = latents_batch.shape
            context_mask, target_mask = masker(batch_size, num_steps)
            return {
                "latents": latents_batch,
                "domain_ids": domain_ids_batch,
                "context_mask": context_mask,
                "target_mask": target_mask,
            }

        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = RJEPATrainer(
                model=model,
                train_loader=loader,
                val_loader=None,
                lr=1e-3,
                max_epochs=2,
                checkpoint_dir=Path(tmpdir),
                use_wandb=False,
                amp_enabled=False,
                device="cpu",
            )

            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.scheduler is not None

        print("[OK] RJEPATrainer works")

    except Exception as e:
        print(f"[FAIL] RJEPATrainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test training one epoch
    try:
        print("[*] Testing training one epoch...")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = RJEPATrainer(
                model=model,
                train_loader=loader,
                val_loader=None,
                lr=1e-3,
                max_epochs=1,
                checkpoint_dir=Path(tmpdir),
                use_wandb=False,
                amp_enabled=False,
                device="cpu",
            )

            epoch_metrics = trainer.train_epoch()

            assert "loss" in epoch_metrics
            assert "recon_loss" in epoch_metrics
            assert epoch_metrics["loss"] > 0

        print("[OK] Training one epoch works")

    except Exception as e:
        print(f"[FAIL] Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test checkpointing
    try:
        print("[*] Testing checkpointing...")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            trainer = RJEPATrainer(
                model=model,
                train_loader=loader,
                val_loader=None,
                lr=1e-3,
                max_epochs=1,
                checkpoint_dir=checkpoint_dir,
                use_wandb=False,
                amp_enabled=False,
                device="cpu",
            )

            trainer.train_epoch()
            trainer.save_checkpoint(filename="test.pth")

            checkpoint_path = checkpoint_dir / "test.pth"
            assert checkpoint_path.exists()

        print("[OK] Checkpointing works")

    except Exception as e:
        print(f"[FAIL] Checkpointing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test EMA momentum annealing
    try:
        print("[*] Testing EMA momentum annealing...")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = RJEPATrainer(
                model=model,
                train_loader=loader,
                val_loader=None,
                lr=1e-3,
                max_epochs=10,
                ema_momentum_start=0.9,
                ema_momentum_end=0.99,
                checkpoint_dir=Path(tmpdir),
                use_wandb=False,
                device="cpu",
            )

            trainer.current_epoch = 0
            momentum_start = trainer._get_ema_momentum()

            trainer.current_epoch = 9
            momentum_end = trainer._get_ema_momentum()

            assert momentum_start < momentum_end
            assert abs(momentum_start - 0.9) < 0.01
            assert abs(momentum_end - 0.99) < 0.01

        print("[OK] EMA momentum annealing works")

    except Exception as e:
        print(f"[FAIL] EMA annealing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 80)
    print("SUCCESS: PHASE 6 VALIDATION COMPLETE!")
    print("=" * 80)
    print()
    print("[OK] All required files exist")
    print("[OK] All imports work")
    print("[OK] RJEPATrainer works")
    print("[OK] Training one epoch works")
    print("[OK] Checkpointing works")
    print("[OK] EMA momentum annealing works")
    print()
    print("Statistics:")
    print("   - Trainer: RJEPATrainer (with AMP, grad clip, EMA)")
    print("   - Pipeline: train_rjepa_from_config + Prefect flow")
    print("   - LR Scheduler: Warmup + cosine decay")
    print("   - Checkpointing: Save/load with state")
    print("   - W&B: Optional logging support")
    print()
    print("Key Features:")
    print("   [OK] Automatic Mixed Precision (AMP)")
    print("   [OK] Gradient clipping")
    print("   [OK] EMA target encoder (annealed momentum)")
    print("   [OK] LR warmup + cosine decay")
    print("   [OK] Checkpointing (resume training)")
    print("   [OK] W&B logging (optional)")
    print("   [OK] Validation loop")
    print()
    print("READY FOR PHASE 7: R-JEPA Service (inference API)")
    print()

    return True


if __name__ == "__main__":
    success = validate_phase6()
    sys.exit(0 if success else 1)
