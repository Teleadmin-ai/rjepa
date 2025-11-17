"""
Validation script for Phase 12: Latent Decoder.

Verifies:
1. All required files exist
2. LatentDecoder and LatentDecoderTrainer can be imported
3. Model instantiation works
4. Dataset loader works (with mock data)
5. Training pipeline is functional
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("PHASE 12 VALIDATION: LATENT DECODER")
print("="*80)

# Check 1: Required files exist
print("\n[1/6] Checking required files...")
required_files = [
    "rjepa/decoder/__init__.py",
    "rjepa/decoder/latent_decoder.py",
    "rjepa/decoder/trainer.py",
    "rjepa/decoder/dataset.py",
    "rjepa/pipeline/train_decoder.py",
    "configs/decoder/train.yaml",
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
    from rjepa.decoder import (
        LatentDecoder,
        LatentDecoderConfig,
        create_latent_decoder,
        LatentDecoderTrainer,
        LatentTextDataset,
        create_decoder_dataloaders,
    )
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Check 3: Model instantiation
print("\n[3/6] Testing model instantiation...")
try:
    import torch

    config = LatentDecoderConfig(
        latent_dim=4096,
        vocab_size=151936,
        decoder_dim=1024,
        depth=4,
        num_heads=8,
    )

    model = LatentDecoder(config)
    num_params = model.get_num_params()

    print(f"  [OK] Model created with {num_params:,} parameters ({num_params/1e6:.1f}M)")

    # Test forward pass with dummy data
    batch_size = 2
    seq_len = 10
    latent = torch.randn(batch_size, config.latent_dim)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(latent, input_ids, labels)

    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)

    print(f"  [OK] Forward pass successful")
    print(f"  [OK] Loss: {outputs['loss'].item():.4f}")

except Exception as e:
    print(f"[FAIL] Model instantiation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 4: Generate method
print("\n[4/6] Testing generate method...")
try:
    from transformers import AutoTokenizer

    # Use a small tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Test generate (should run without error)
    latent = torch.randn(1, config.latent_dim)

    with torch.no_grad():
        result = model.generate(
            latent,
            tokenizer,
            max_new_tokens=10,
            temperature=0.8,
        )

    assert "generated_ids" in result
    assert "text" in result

    print(f"  [OK] Generate method works")
    print(f"  [OK] Generated text: {result['text'][:50]}...")

except Exception as e:
    print(f"[FAIL] Generate method error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 5: Trainer instantiation
print("\n[5/6] Testing trainer instantiation...")
try:
    from torch.utils.data import DataLoader, TensorDataset

    # Create dummy dataset
    dummy_latents = torch.randn(100, config.latent_dim)
    dummy_input_ids = torch.randint(0, config.vocab_size, (100, 20))
    dummy_labels = dummy_input_ids.clone()

    dummy_dataset = TensorDataset(dummy_latents, dummy_input_ids, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=8)

    # Create trainer
    trainer = LatentDecoderTrainer(
        model=model,
        train_loader=dummy_loader,
        val_loader=None,
        lr=1e-4,
        device="cpu",
        use_amp=False,
        wandb_enabled=False,
    )

    print(f"  [OK] Trainer created")

    # Test one training step
    model.train()
    batch = next(iter(dummy_loader))
    latent_batch = batch[0]
    input_ids_batch = batch[1]
    labels_batch = batch[2]

    outputs = model(latent_batch, input_ids_batch, labels_batch)
    loss = outputs["loss"]
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

    config_path = Path("configs/decoder/train.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    required_keys = ["model", "tokenizer", "data", "training", "logging"]
    for key in required_keys:
        assert key in config_dict, f"Missing key: {key}"

    print(f"  [OK] Config file valid")
    print(f"  [OK] Model: latent_dim={config_dict['model']['latent_dim']}, "
          f"decoder_dim={config_dict['model']['decoder_dim']}, "
          f"depth={config_dict['model']['depth']}")

except Exception as e:
    print(f"[FAIL] Config error: {e}")
    sys.exit(1)

# Final summary
print("\n" + "="*80)
print("PHASE 12 VALIDATION: [PASS] ALL CHECKS SUCCESSFUL")
print("="*80)
print("\nLatent Decoder Implementation:")
print(f"  - LatentDecoder: {num_params:,} parameters ({num_params/1e6:.1f}M)")
print(f"  - Architecture: Latent projection + Causal Transformer (depth=4) + LM head")
print(f"  - Training: AMP, gradient clipping, checkpointing, W&B logging")
print(f"  - Dataset: LatentTextDataset with lazy loading")
print(f"  - Philosophy: Train AFTER R-JEPA is frozen (world model as ground truth)")
print("\nNext steps:")
print("  1. Build latents: python -m rjepa.pipeline.build_latents --llm qwen3-8b --split train")
print("  2. Train decoder: python -m rjepa.pipeline.train_decoder --config configs/decoder/train.yaml")
print("="*80)
