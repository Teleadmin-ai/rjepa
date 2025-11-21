"""
Test forward pass with ORIGINAL model configuration.

This script tests the exact same pipeline as train_rjepa but with detailed timing and error handling.
"""

import sys
import time
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("TEST FORWARD PASS - ORIGINAL MODEL")
print("=" * 80)
print()

# 1. Imports
print("[1/8] Importing modules...", flush=True)
start = time.time()
import yaml
from rjepa.pipeline.train_rjepa import create_dataloaders, create_rjepa_model
from rjepa.jepa.maskers import create_masker, MaskCollator
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 2. Load config
print("[2/8] Loading config...", flush=True)
start = time.time()
config_path = project_root / "configs" / "rjepa" / "train.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
print(f"  Config loaded:", flush=True)
print(f"    - encoder_embed_dim: {config['model']['encoder_embed_dim']}")
print(f"    - depth_encoder: {config['model']['depth_encoder']}")
print(f"    - batch_size: {config['training']['batch_size']}")
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 3. Create dataloaders
print("[3/8] Creating dataloaders...", flush=True)
start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader, masker_config = create_dataloaders(
    train_latents_dir=Path(config["data"]["train_latents_dir"]),
    val_latents_dir=Path(config["data"].get("val_latents_dir")),
    batch_size=config["training"]["batch_size"],
    num_workers=0,  # Force 0 workers for debugging
    masker_config=config.get("masker", {"type": "contiguous"}),
    device=device,
)
print(f"  Train samples: {len(train_loader.dataset)}", flush=True)
print(f"  Train batches: {len(train_loader)}", flush=True)
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 4. Create model
print("[4/8] Creating R-JEPA model...", flush=True)
start = time.time()
model = create_rjepa_model(config["model"])
model = model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"  Total params: {num_params:,}", flush=True)
print(f"  Device: {device}", flush=True)
if device == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"  CUDA memory allocated: {allocated:.2f} GB", flush=True)
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 5. Create masker
print("[5/8] Creating MaskCollator...", flush=True)
start = time.time()
masker = create_masker(masker_config)
mask_collator = MaskCollator(masker, device="cpu")  # Collate on CPU first
print(f"  Masker type: {masker_config.get('type', 'contiguous')}", flush=True)
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 6. Get one batch and apply masking
print("[6/8] Getting batch and applying masking...", flush=True)
start = time.time()
batch = next(iter(train_loader))
print(f"  Raw batch type: {type(batch)}", flush=True)
print(f"  Raw batch length: {len(batch)}", flush=True)
if isinstance(batch, list) and len(batch) > 0:
    print(f"  First item type: {type(batch[0])}", flush=True)
    if isinstance(batch[0], tuple) and len(batch[0]) >= 1:
        print(f"  First H shape: {batch[0][0].shape}", flush=True)

# Apply MaskCollator
masked_batch = mask_collator(batch)
print(f"  Masked batch keys: {list(masked_batch.keys())}", flush=True)
latents = masked_batch["latents"]
context_mask_bool = masked_batch["context_mask"]
target_mask_bool = masked_batch["target_mask"]
domain_ids = masked_batch.get("domain_ids")
print(f"  Latents shape: {latents.shape}", flush=True)
print(f"  Context mask shape: {context_mask_bool.shape}", flush=True)
print(f"  Target mask shape: {target_mask_bool.shape}", flush=True)
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 7. Move to GPU and prepare masks
print("[7/8] Moving to GPU and preparing masks...", flush=True)
start = time.time()
latents = latents.to(device)
context_mask_bool = context_mask_bool.to(device)
target_mask_bool = target_mask_bool.to(device)
if domain_ids is not None:
    domain_ids = domain_ids.to(device)

# Convert boolean masks to index masks for V-JEPA format
B = latents.size(0)
context_mask = []
target_mask = []
for b in range(B):
    ctx_idx = (context_mask_bool[b]).nonzero(as_tuple=False).squeeze(-1)
    tgt_idx = (target_mask_bool[b]).nonzero(as_tuple=False).squeeze(-1)
    context_mask.append(ctx_idx.unsqueeze(0))
    target_mask.append(tgt_idx.unsqueeze(0))

# Stack into tensors (pad if needed)
max_ctx = max(m.size(1) for m in context_mask)
max_tgt = max(m.size(1) for m in target_mask)

context_mask_tensor = torch.zeros(B, max_ctx, dtype=torch.long, device=device)
target_mask_tensor = torch.zeros(B, max_tgt, dtype=torch.long, device=device)

for b in range(B):
    ctx_len = context_mask[b].size(1)
    tgt_len = target_mask[b].size(1)
    context_mask_tensor[b, :ctx_len] = context_mask[b]
    target_mask_tensor[b, :tgt_len] = target_mask[b]

# Wrap in list for V-JEPA format
context_mask = [context_mask_tensor]
target_mask = [target_mask_tensor]

print(f"  Latents on {latents.device}: {latents.shape}", flush=True)
print(f"  Context mask: {context_mask_tensor.shape}", flush=True)
print(f"  Target mask: {target_mask_tensor.shape}", flush=True)
if device == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"  CUDA memory allocated: {allocated:.2f} GB", flush=True)
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 8. Forward pass
print("[8/8] Testing forward pass...", flush=True)
start = time.time()
model.eval()
try:
    with torch.no_grad():
        outputs = model(
            latents,
            context_mask,
            target_mask,
            domain_ids,
            compute_loss=True,
        )
    print(f"  Loss: {outputs['loss'].item():.4f}", flush=True)
    print(f"  Recon loss: {outputs['recon_loss'].item():.4f}", flush=True)
    print(f"  Var reg loss: {outputs['var_reg_loss'].item():.4f}", flush=True)
    print(f"  z_pred shape: {outputs['z_pred'].shape}", flush=True)
    print(f"  z_target shape: {outputs['z_target'].shape}", flush=True)
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

if device == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  CUDA memory allocated: {allocated:.2f} GB", flush=True)
    print(f"  CUDA memory reserved: {reserved:.2f} GB", flush=True)
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

print("=" * 80)
print("FORWARD PASS SUCCESSFUL!")
print("=" * 80)
