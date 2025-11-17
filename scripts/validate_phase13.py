"""
Validation script for Phase 13: Logit Guidance.

Verifies:
1. All required files exist
2. LogitGuidance and trainer can be imported
3. Model instantiation works
4. Guidance application works
5. Training pipeline is functional
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("PHASE 13 VALIDATION: LOGIT GUIDANCE")
print("="*80)

# Check 1: Required files exist
print("\n[1/6] Checking required files...")
required_files = [
    "rjepa/inference/logit_guidance.py",
    "rjepa/inference/logit_guidance_trainer.py",
    "configs/guidance/train.yaml",
]

missing_files = []
for file_path in required_files:
    full_path = Path(file_path)
    if not full_path.exists():
        missing_files.append(file_path)
        print(f"  [X] Missing: {file_path}")
    else:
        print(f"  [OK] Found: {file_path}")

if missing_files:
    print(f"\n[FAIL] Missing {len(missing_files)} required files")
    sys.exit(1)

print("[OK] All required files exist")

# Check 2: Imports
print("\n[2/6] Testing imports...")
try:
    from rjepa.inference import (
        LogitGuidance,
        LogitGuidanceConfig,
        create_logit_guidance,
        guided_generation_step,
        generate_with_guidance,
        LogitGuidanceTrainer,
    )
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Check 3: Model instantiation
print("\n[3/6] Testing LogitGuidance instantiation...")
try:
    import torch

    config = LogitGuidanceConfig(
        latent_dim=4096,
        vocab_size=151936,
        hidden_dim=2048,
        alpha=0.3,
    )

    guidance = LogitGuidance(config)
    num_params = sum(p.numel() for p in guidance.parameters())

    print(f"  [OK] LogitGuidance created with {num_params:,} parameters ({num_params/1e6:.1f}M)")

    # Test forward pass
    batch_size = 4
    latent = torch.randn(batch_size, config.latent_dim)

    logit_bias = guidance(latent)

    assert logit_bias.shape == (batch_size, config.vocab_size)
    print(f"  [OK] Forward pass successful")
    print(f"  [OK] Logit bias shape: {logit_bias.shape}")

except Exception as e:
    print(f"[FAIL] Model instantiation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 4: Guidance application
print("\n[4/6] Testing apply_guidance...")
try:
    # Test guidance application
    llm_logits = torch.randn(batch_size, config.vocab_size)

    guided_logits = guidance.apply_guidance(
        llm_logits=llm_logits,
        latent=latent,
        alpha=0.3,
    )

    assert guided_logits.shape == llm_logits.shape
    assert not torch.allclose(guided_logits, llm_logits)  # Should be different

    # Check that bias was applied
    delta = (guided_logits - llm_logits).abs().mean().item()
    print(f"  [OK] Guidance applied (mean delta: {delta:.4f})")

    # Test with different alpha
    guided_logits_alpha0 = guidance.apply_guidance(
        llm_logits=llm_logits,
        latent=latent,
        alpha=0.0,
    )

    assert torch.allclose(guided_logits_alpha0, llm_logits, atol=1e-5)
    print(f"  [OK] Alpha=0 returns original logits (no guidance)")

except Exception as e:
    print(f"[FAIL] apply_guidance error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 5: Trainer instantiation
print("\n[5/6] Testing LogitGuidanceTrainer...")
try:
    from torch.utils.data import DataLoader, TensorDataset

    # Create dummy dataset
    # Format: (latent, llm_logits, true_token)
    dummy_latents = torch.randn(100, config.latent_dim)
    dummy_logits = torch.randn(100, config.vocab_size)
    dummy_tokens = torch.randint(0, config.vocab_size, (100,))

    # Wrap in dict format
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, latents, logits, tokens):
            self.latents = latents
            self.logits = logits
            self.tokens = tokens

        def __len__(self):
            return len(self.latents)

        def __getitem__(self, idx):
            return {
                "latent": self.latents[idx],
                "llm_logits": self.logits[idx],
                "true_token": self.tokens[idx],
            }

    dummy_dataset = DummyDataset(dummy_latents, dummy_logits, dummy_tokens)
    dummy_loader = DataLoader(dummy_dataset, batch_size=16)

    # Create trainer
    trainer = LogitGuidanceTrainer(
        guidance=guidance,
        train_loader=dummy_loader,
        val_loader=None,
        lr=3e-4,
        device="cpu",
        use_amp=False,
        wandb_enabled=False,
    )

    print(f"  [OK] Trainer created")

    # Test one training step
    guidance.train()
    batch = next(iter(dummy_loader))
    latent_batch = batch["latent"]
    llm_logits_batch = batch["llm_logits"]
    true_token_batch = batch["true_token"]

    guided_logits_batch = guidance.apply_guidance(
        llm_logits=llm_logits_batch,
        latent=latent_batch,
    )

    loss = torch.nn.functional.cross_entropy(guided_logits_batch, true_token_batch)
    loss.backward()

    print(f"  [OK] Training step successful (loss: {loss.item():.4f})")

except Exception as e:
    print(f"[FAIL] Trainer error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 6: Config file validity
print("\n[6/6] Testing config file...")
try:
    import yaml

    config_path = Path("configs/guidance/train.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    required_keys = ["model", "data", "training", "logging"]
    for key in required_keys:
        assert key in config_dict, f"Missing key: {key}"

    print(f"  [OK] Config file valid")
    print(f"  [OK] Model: latent_dim={config_dict['model']['latent_dim']}, "
          f"vocab_size={config_dict['model']['vocab_size']}, "
          f"alpha={config_dict['model']['alpha']}")

except Exception as e:
    print(f"[FAIL] Config error: {e}")
    sys.exit(1)

# Final summary
print("\n" + "="*80)
print("PHASE 13 VALIDATION: [PASS] ALL CHECKS SUCCESSFUL")
print("="*80)
print("\nLogit Guidance Implementation:")
print(f"  - LogitGuidance: {num_params:,} parameters ({num_params/1e6:.1f}M)")
print(f"  - Architecture: Latent [4096] -> MLP [2048] -> Vocab [151936]")
print(f"  - Guidance: logits_final = logits_llm + alpha * logit_bias")
print(f"  - Training: KL/CE loss, AMP, gradient clipping")
print(f"  - Philosophy: Works without LLM hidden state access (API-friendly)")
print("\nAdvantages over Nudge mode:")
print("  - Compatible with API-based LLMs (OpenAI, Anthropic, etc.)")
print("  - Less invasive (doesn't modify internal representations)")
print("  - Easier to implement (just add logit bias)")
print("\nNext steps:")
print("  1. Generate guidance dataset: (latent, llm_logits, true_token) tuples")
print("  2. Train guidance: python -m rjepa.pipeline.train_guidance")
print("  3. Use in inference: generate_with_guidance()")
print("="*80)
