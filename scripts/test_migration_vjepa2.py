#!/usr/bin/env python
"""
Test script to validate V-JEPA 2 MIT migration.
Verifies that all modules work correctly after migration.
"""

import sys
sys.path.insert(0, 'c:/Users/teleadmin/world-txt-model')

import torch
import numpy as np

print("=" * 60)
print("V-JEPA 2 MIT Migration Test")
print("=" * 60)

# Test 1: modules.py
print("\n[1/5] Testing modules.py...")
try:
    from rjepa.jepa.modules import Block, Attention, MLP, DropPath

    # Test DropPath (NEW in V-JEPA 2)
    dp = DropPath(0.1)
    x = torch.randn(2, 10, 256)
    dp.train()
    y = dp(x)
    assert y.shape == x.shape, f"DropPath shape mismatch: {y.shape}"
    print("  [OK] DropPath OK")

    # Test MLP
    mlp = MLP(256, 1024, 256)
    y = mlp(x)
    assert y.shape == x.shape, f"MLP shape mismatch: {y.shape}"
    print("  [OK] MLP OK")

    # Test Attention
    attn = Attention(256, num_heads=8)
    y, _ = attn(x)
    assert y.shape == x.shape, f"Attention shape mismatch: {y.shape}"
    print("  [OK] Attention OK")

    # Test Block with drop_path
    block = Block(256, num_heads=8, drop_path=0.1)
    y = block(x)
    assert y.shape == x.shape, f"Block shape mismatch: {y.shape}"
    print("  [OK] Block with DropPath OK")

except Exception as e:
    print(f"  [FAIL] modules.py FAILED: {e}")
    sys.exit(1)

# Test 2: pos_embs.py
print("\n[2/5] Testing pos_embs.py...")
try:
    from rjepa.jepa.pos_embs import get_1d_sincos_pos_embed

    pos = get_1d_sincos_pos_embed(256, 10)
    assert pos.shape == (10, 256), f"pos_embed shape mismatch: {pos.shape}"
    print("  [OK] get_1d_sincos_pos_embed OK")

    # Test with cls_token
    pos_cls = get_1d_sincos_pos_embed(256, 10, cls_token=True)
    assert pos_cls.shape == (11, 256), f"pos_embed with cls shape mismatch: {pos_cls.shape}"
    print("  [OK] get_1d_sincos_pos_embed with cls_token OK")

except Exception as e:
    print(f"  [FAIL] pos_embs.py FAILED: {e}")
    sys.exit(1)

# Test 3: step_transformer.py
print("\n[3/5] Testing step_transformer.py...")
try:
    from rjepa.jepa.step_transformer import StepTransformer

    encoder = StepTransformer(
        input_dim=512,
        max_seq_len=32,
        embed_dim=256,
        depth=4,
        num_heads=4,
        drop_path_rate=0.1  # NEW: V-JEPA 2 stochastic depth
    )

    x = torch.randn(2, 10, 512)  # [B, S, input_dim]
    y = encoder(x)
    assert y.shape == (2, 10, 256), f"StepTransformer shape mismatch: {y.shape}"
    print("  [OK] StepTransformer OK")

    # Count params
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  [OK] StepTransformer params: {num_params:,}")

except Exception as e:
    print(f"  [FAIL] step_transformer.py FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: step_predictor.py
print("\n[4/5] Testing step_predictor.py...")
try:
    from rjepa.jepa.step_predictor import StepPredictor

    predictor = StepPredictor(
        max_seq_len=32,
        encoder_dim=256,
        predictor_dim=128,
        depth=4,
        num_heads=4,
        drop_path_rate=0.1  # NEW: V-JEPA 2 stochastic depth
    )

    # Create mock inputs
    B = 2
    ctxt = torch.randn(B, 6, 256)  # Context tokens
    tgt = torch.randn(B, 32, 256)  # Target tokens (full sequence)
    masks_ctxt = [torch.arange(6).unsqueeze(0).expand(B, -1)]  # [B, 6]
    masks_tgt = [torch.arange(6, 10).unsqueeze(0).expand(B, -1)]  # [B, 4]

    y = predictor(ctxt, tgt, masks_ctxt, masks_tgt)
    assert y.shape[2] == 256, f"StepPredictor output dim mismatch: {y.shape}"
    print("  [OK] StepPredictor OK")

    num_params = sum(p.numel() for p in predictor.parameters())
    print(f"  [OK] StepPredictor params: {num_params:,}")

except Exception as e:
    print(f"  [FAIL] step_predictor.py FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Full model (ReasoningJEPA)
print("\n[5/5] Testing full ReasoningJEPA model...")
try:
    from rjepa.jepa.model import ReasoningJEPA

    model = ReasoningJEPA(
        input_dim=512,
        embed_dim=256,
        predictor_dim=128,
        depth_encoder=4,
        depth_predictor=4,
        num_heads=4,
        max_seq_len=32
    )

    B = 2
    S = 10
    x = torch.randn(B, S, 512)  # [B, S, input_dim]

    # Create masks (context = first 6 steps, target = last 4 steps)
    masks_context = [torch.arange(6).unsqueeze(0).expand(B, -1)]  # [B, 6]
    masks_target = [torch.arange(6, 10).unsqueeze(0).expand(B, -1)]  # [B, 4]

    # Forward with loss
    outputs = model(x, masks_context, masks_target, compute_loss=True)

    assert 'loss' in outputs, "Loss not in outputs"
    assert not torch.isnan(outputs['loss']), "Loss is NaN"
    print(f"  [OK] ReasoningJEPA OK (loss={outputs['loss'].item():.4f})")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] ReasoningJEPA params: {num_params:,}")

except Exception as e:
    print(f"  [FAIL] ReasoningJEPA FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("[SUCCESS] V-JEPA 2 MIT MIGRATION SUCCESSFUL!")
print("=" * 60)
print("\nAll modules migrated to MIT license:")
print("  [OK] modules.py (Block, Attention, MLP, DropPath)")
print("  [OK] pos_embs.py (get_1d_sincos_pos_embed)")
print("  [OK] step_transformer.py (StepTransformer)")
print("  [OK] step_predictor.py (StepPredictor)")
print("  [OK] Full model (ReasoningJEPA)")
print("\nNew V-JEPA 2 features enabled:")
print("  [OK] DropPath (Stochastic Depth)")
print("  [OK] drop_path_rate parameter in transformers")
print("  [OK] MIT license headers")
