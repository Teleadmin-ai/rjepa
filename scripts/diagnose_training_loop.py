"""
Diagnostic précis de la boucle de training R-JEPA.
Teste chaque étape pour identifier le blocage exact.
"""

import sys
import time
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_flush(msg=""):
    print(msg, flush=True)

print_flush("=" * 80)
print_flush("DIAGNOSTIC TRAINING LOOP - STEP BY STEP")
print_flush("=" * 80)
print_flush()

# 1. Imports
print_flush("[1/12] Importing modules...")
start = time.time()
import yaml
from rjepa.pipeline.train_rjepa import create_dataloaders, create_rjepa_model
from rjepa.jepa.maskers import create_masker, MaskCollator
from torch.cuda.amp import GradScaler, autocast
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

# 2. Load config
print_flush("[2/12] Loading config...")
start = time.time()
config_path = project_root / "configs" / "rjepa" / "train.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
print_flush(f"  encoder_embed_dim: {config['model']['encoder_embed_dim']}")
print_flush(f"  depth_encoder: {config['model']['depth_encoder']}")
print_flush(f"  batch_size: {config['training']['batch_size']}")
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

# 3. Create dataloaders
print_flush("[3/12] Creating dataloaders...")
start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader, masker_config = create_dataloaders(
    train_latents_dir=Path(config["data"]["train_latents_dir"]),
    val_latents_dir=Path(config["data"].get("val_latents_dir")),
    batch_size=config["training"]["batch_size"],
    num_workers=0,  # Force 0 workers
    masker_config=config.get("masker", {"type": "contiguous"}),
    device=device,
)
print_flush(f"  Train samples: {len(train_loader.dataset)}")
print_flush(f"  Train batches: {len(train_loader)}")
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

# 4. Create model
print_flush("[4/12] Creating R-JEPA model...")
start = time.time()
model = create_rjepa_model(config["model"])
model = model.to(device)
model.train()
num_params = sum(p.numel() for p in model.parameters())
print_flush(f"  Total params: {num_params:,}")
print_flush(f"  Device: {device}")
if device == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**3
    print_flush(f"  CUDA memory allocated: {allocated:.2f} GB")
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

# 5. Create optimizer
print_flush("[5/12] Creating optimizer...")
start = time.time()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["training"]["lr"],
    weight_decay=config["training"]["weight_decay"],
    betas=(0.9, 0.95),
)
print_flush(f"  LR: {config['training']['lr']}")
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

# 6. Create scaler
print_flush("[6/12] Creating AMP scaler...")
start = time.time()
scaler = GradScaler() if config["training"].get("amp_enabled", True) else None
print_flush(f"  AMP enabled: {scaler is not None}")
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

# 7. Create masker
print_flush("[7/12] Creating MaskCollator...")
start = time.time()
masker = create_masker(masker_config)
mask_collator = MaskCollator(masker, device="cpu")  # CPU first
print_flush(f"  Masker type: {masker_config.get('type', 'contiguous')}")
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

# 8. Get batch from dataloader
print_flush("[8/12] Getting batch from DataLoader...")
start = time.time()
batch_iter = iter(train_loader)
batch = next(batch_iter)
print_flush(f"  Raw batch type: {type(batch)}")
print_flush(f"  Raw batch length: {len(batch)}")
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

# 9. Apply mask collator
print_flush("[9/12] Applying MaskCollator...")
start = time.time()
masked_batch = mask_collator(batch)
latents = masked_batch["latents"]
context_mask_bool = masked_batch["context_mask"]
target_mask_bool = masked_batch["target_mask"]
domain_ids = masked_batch.get("domain_ids")
print_flush(f"  Latents shape: {latents.shape}")
print_flush(f"  Context mask shape: {context_mask_bool.shape}")
print_flush(f"  Target mask shape: {target_mask_bool.shape}")
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

# 10. Move to GPU and convert masks
print_flush("[10/12] Moving to GPU and converting masks...")
start = time.time()
latents = latents.to(device)
context_mask_bool = context_mask_bool.to(device)
target_mask_bool = target_mask_bool.to(device)
if domain_ids is not None:
    domain_ids = domain_ids.to(device)

# Convert boolean masks to index masks
B = latents.size(0)
context_mask = []
target_mask = []
for b in range(B):
    ctx_idx = (context_mask_bool[b]).nonzero(as_tuple=False).squeeze(-1)
    tgt_idx = (target_mask_bool[b]).nonzero(as_tuple=False).squeeze(-1)
    context_mask.append(ctx_idx.unsqueeze(0))
    target_mask.append(tgt_idx.unsqueeze(0))

max_ctx = max(m.size(1) for m in context_mask)
max_tgt = max(m.size(1) for m in target_mask)

context_mask_tensor = torch.zeros(B, max_ctx, dtype=torch.long, device=device)
target_mask_tensor = torch.zeros(B, max_tgt, dtype=torch.long, device=device)

for b in range(B):
    ctx_len = context_mask[b].size(1)
    tgt_len = target_mask[b].size(1)
    context_mask_tensor[b, :ctx_len] = context_mask[b]
    target_mask_tensor[b, :tgt_len] = target_mask[b]

context_mask = [context_mask_tensor]
target_mask = [target_mask_tensor]

print_flush(f"  Context mask tensor: {context_mask_tensor.shape}")
print_flush(f"  Target mask tensor: {target_mask_tensor.shape}")
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

# 11. Forward pass
print_flush("[11/12] Forward pass with AMP...")
start = time.time()
optimizer.zero_grad()
with autocast():
    outputs = model(
        latents,
        context_mask,
        target_mask,
        domain_ids,
        compute_loss=True,
    )
    loss = outputs["loss"]
print_flush(f"  Loss: {loss.item():.4f}")
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

# 12. Backward pass
print_flush("[12/12] Backward pass and optimizer step...")
start = time.time()
scaler.scale(loss).backward()
print_flush(f"  Backward done")
scaler.unscale_(optimizer)
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print_flush(f"  Grad norm: {grad_norm:.4f}")
scaler.step(optimizer)
scaler.update()
print_flush(f"  Optimizer step done")
print_flush(f"  Done ({time.time() - start:.2f}s)")
print_flush()

if device == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print_flush(f"Final CUDA memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

print_flush()
print_flush("=" * 80)
print_flush("DIAGNOSTIC COMPLETE - ALL STEPS SUCCESSFUL!")
print_flush("The training loop should work. Problem is likely in:")
print_flush("1. rich.progress.track not flushing to log file")
print_flush("2. Multiple batches causing issues")
print_flush("=" * 80)

# Test 3 more batches
print_flush()
print_flush("Testing 3 more batches...")
for i in range(3):
    start = time.time()
    batch = next(batch_iter)
    masked_batch = mask_collator(batch)
    latents = masked_batch["latents"].to(device)
    context_mask_bool = masked_batch["context_mask"].to(device)
    target_mask_bool = masked_batch["target_mask"].to(device)

    # Quick mask conversion
    B = latents.size(0)
    ctx_idx = [(context_mask_bool[b]).nonzero(as_tuple=False).squeeze(-1).unsqueeze(0) for b in range(B)]
    tgt_idx = [(target_mask_bool[b]).nonzero(as_tuple=False).squeeze(-1).unsqueeze(0) for b in range(B)]

    max_ctx = max(m.size(1) for m in ctx_idx)
    max_tgt = max(m.size(1) for m in tgt_idx)

    ctx_tensor = torch.zeros(B, max_ctx, dtype=torch.long, device=device)
    tgt_tensor = torch.zeros(B, max_tgt, dtype=torch.long, device=device)
    for b in range(B):
        ctx_tensor[b, :ctx_idx[b].size(1)] = ctx_idx[b]
        tgt_tensor[b, :tgt_idx[b].size(1)] = tgt_idx[b]

    optimizer.zero_grad()
    with autocast():
        outputs = model(latents, [ctx_tensor], [tgt_tensor], None, compute_loss=True)
        loss = outputs["loss"]
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print_flush(f"  Batch {i+2}: loss={loss.item():.4f}, time={time.time()-start:.2f}s")

print_flush()
print_flush("=" * 80)
print_flush("ALL 4 BATCHES COMPLETED SUCCESSFULLY!")
print_flush("=" * 80)
