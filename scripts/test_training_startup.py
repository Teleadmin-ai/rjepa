"""
Script de debugging pour identifier où le training se bloque.
"""
import sys
import torch
from pathlib import Path

print("=" * 80)
print("DEBUGGING TRAINING STARTUP")
print("=" * 80)

# 1. Test import
print("\n[1] Testing imports...")
try:
    from rjepa.jepa.dataset import LatentDatasetMultiShard
    from rjepa.jepa.maskers import create_masker, MaskCollator, simple_collate
    from rjepa.jepa.model import ReasoningJEPA
    from rjepa.jepa.trainer import RJEPATrainer
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# 2. Test dataset loading
print("\n[2] Testing dataset loading...")
try:
    latents_dir = Path("data/latents/qwen3-8b/academic_shards")
    dataset = LatentDatasetMultiShard(latents_dir, device="cpu")
    print(f"✅ Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Test single sample
print("\n[3] Testing single sample retrieval...")
try:
    sample = dataset[0]
    latents, domain_id = sample
    print(f"✅ Sample retrieved: latents shape={latents.shape}, domain_id={domain_id}")
except Exception as e:
    print(f"❌ Sample retrieval failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test DataLoader
print("\n[4] Testing DataLoader with batch_size=1...")
try:
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate,
    )
    print(f"✅ DataLoader created: {len(loader)} batches")
except Exception as e:
    print(f"❌ DataLoader creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Test first batch
print("\n[5] Testing first batch retrieval...")
try:
    batch = next(iter(loader))
    print(f"✅ First batch retrieved: type={type(batch)}, len={len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}")
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        print(f"   First item: type={type(batch[0])}, len={len(batch[0]) if isinstance(batch[0], (list, tuple)) else 'N/A'}")
except Exception as e:
    print(f"❌ Batch retrieval failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Test masker
print("\n[6] Testing masker on batch...")
try:
    masker_config = {
        "type": "contiguous",
        "min_mask_ratio": 0.3,
        "max_mask_ratio": 0.7,
        "num_blocks": 1,
    }
    masker = create_masker(masker_config)
    mask_collator = MaskCollator(masker, device="cpu")
    masked_batch = mask_collator(batch)
    print(f"✅ Masker applied: masked_batch keys={list(masked_batch.keys())}")
    print(f"   latents: {masked_batch['latents'].shape}")
    print(f"   context_mask: {masked_batch['context_mask'].shape}")
    print(f"   target_mask: {masked_batch['target_mask'].shape}")
except Exception as e:
    print(f"❌ Masker failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Test model creation
print("\n[7] Testing model creation...")
try:
    model = ReasoningJEPA(
        dim=4096,
        encoder_embed_dim=2048,
        depth_encoder=6,
        depth_predictor=4,
        num_heads=8,
        predictor_dim=1024,
        mlp_ratio=4.0,
        dropout=0.0,
        max_steps=512,
        domain_embed_dim=64,
        num_domains=50,
        ema_momentum=0.996,
        loss_type="l1",
        var_reg_weight=0.01,
        var_reg_target=1.0,
        contrastive_weight=0.1,
        contrastive_temperature=0.07,
        use_hard_negatives=True,
    )
    print(f"✅ Model created")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 8. Test model forward pass
print("\n[8] Testing model forward pass...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")

    model = model.to(device)
    latents = masked_batch["latents"].to(device)
    context_mask = masked_batch["context_mask"].to(device)
    target_mask = masked_batch["target_mask"].to(device)
    domain_ids = masked_batch.get("domain_ids")
    if domain_ids is not None:
        domain_ids = domain_ids.to(device)

    print(f"   Forward pass...")
    outputs = model(
        latents,
        context_mask=context_mask,
        target_mask=target_mask,
        domain_ids=domain_ids,
        compute_loss=True,
    )

    print(f"✅ Forward pass successful!")
    print(f"   Loss: {outputs['loss'].item():.6f}")
    print(f"   Recon loss: {outputs['recon_loss'].item():.6f}")
    print(f"   Var reg loss: {outputs['var_reg_loss'].item():.6f}")

except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED! Training should work.")
print("=" * 80)
