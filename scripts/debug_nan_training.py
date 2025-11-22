"""
Debug script to find exactly where NaN appears in training
"""
import sys
import torch
import torch.nn as nn
sys.path.insert(0, "c:/Users/teleadmin/world-txt-model")

from rjepa.jepa.model import ReasoningJEPA, create_rjepa_model
from rjepa.jepa.dataset import LatentDatasetMultiShard
from rjepa.jepa.maskers import ContiguousMasker
from pathlib import Path
import yaml

print("=" * 60)
print("DEBUG: Finding NaN source in training")
print("=" * 60)

# Load config
with open("configs/rjepa/train.yaml") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Create fresh model
print("\n[1] Creating fresh model...")
model = create_rjepa_model(config['model'])
model = model.to(device)

# Check model weights
total_params = 0
nan_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    nan_count = torch.isnan(param).sum().item()
    nan_params += nan_count
    if nan_count > 0:
        print(f"  [NaN] {name}: {nan_count}/{param.numel()}")

print(f"Model initialized: {nan_params}/{total_params} NaN ({100*nan_params/total_params:.2f}%)")
if nan_params > 0:
    print("  ** BUG: Model has NaN at initialization! **")
    sys.exit(1)

# 2. Load one batch of data
print("\n[2] Loading data...")
dataset = LatentDatasetMultiShard(
    latents_dir=Path(config['data']['train_latents_dir']),
)
print(f"Dataset size: {len(dataset)} samples")

# Get first sample
latents, domain_id = dataset[0]
print(f"Sample shape: {latents.shape}")
print(f"Sample stats: min={latents.min():.2f}, max={latents.max():.2f}, mean={latents.mean():.2f}")
print(f"Sample NaN: {torch.isnan(latents).sum().item()}")

if torch.isnan(latents).any():
    print("  ** BUG: Data has NaN! **")
    sys.exit(1)

# 3. Create batch
print("\n[3] Creating batch...")
latents = latents.unsqueeze(0).to(device)  # [1, S, D]
domain_ids = torch.tensor([domain_id], device=device)
print(f"Batch shape: {latents.shape}")

# 4. Create masks (THIS IS WHERE IT MIGHT GO WRONG)
print("\n[4] Creating masks...")
B, S, D = latents.shape
print(f"B={B}, S={S}, D={D}")

masker = ContiguousMasker(
    min_mask_ratio=config['training'].get('min_mask_ratio', 0.3),
    max_mask_ratio=config['training'].get('max_mask_ratio', 0.7)
)
context_mask_bool, target_mask_bool = masker(B, S)
print(f"Context mask shape: {context_mask_bool.shape}")
print(f"Target mask shape: {target_mask_bool.shape}")
print(f"Context visible steps: {context_mask_bool[0].sum().item()}")
print(f"Target masked steps: {target_mask_bool[0].sum().item()}")

# Convert to indices (like trainer does)
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

masks_context = [context_mask_tensor]
masks_target = [target_mask_tensor]

print(f"Context indices: {masks_context[0]}")
print(f"Target indices: {masks_target[0]}")

# 5. Forward pass WITHOUT autocast
print("\n[5] Forward pass WITHOUT AMP...")
model.train()
with torch.no_grad():
    outputs = model(latents, masks_context, masks_target, compute_loss=True)

print(f"z_context shape: {outputs['z_context'].shape}")
print(f"z_target shape: {outputs['z_target'].shape}")
print(f"z_pred shape: {outputs['z_pred'].shape}")
print(f"Loss: {outputs['loss'].item()}")
print(f"Recon loss: {outputs['recon_loss'].item()}")
print(f"Var reg loss: {outputs['var_reg_loss'].item()}")

if torch.isnan(outputs['loss']):
    print("\n  ** BUG: NaN in forward WITHOUT AMP! **")
    print(f"  z_context NaN: {torch.isnan(outputs['z_context']).sum().item()}")
    print(f"  z_target NaN: {torch.isnan(outputs['z_target']).sum().item()}")
    print(f"  z_pred NaN: {torch.isnan(outputs['z_pred']).sum().item()}")
    sys.exit(1)

# 6. Forward pass WITH autocast
print("\n[6] Forward pass WITH AMP...")
from torch.cuda.amp import autocast
with torch.no_grad():
    with autocast():
        outputs_amp = model(latents, masks_context, masks_target, compute_loss=True)

print(f"Loss (AMP): {outputs_amp['loss'].item()}")
print(f"Recon loss (AMP): {outputs_amp['recon_loss'].item()}")
print(f"Var reg loss (AMP): {outputs_amp['var_reg_loss'].item()}")

if torch.isnan(outputs_amp['loss']):
    print("\n  ** BUG: NaN appears WITH AMP! **")
    print("  This is likely an AMP precision issue.")
    sys.exit(1)

# 7. Full training step with backward
print("\n[7] Full training step with backward...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

optimizer.zero_grad()
with autocast():
    outputs_train = model(latents, masks_context, masks_target, compute_loss=True)
    loss = outputs_train['loss']

print(f"Loss before backward: {loss.item()}")

scaler.scale(loss).backward()

# Check gradients
nan_grads = 0
total_grads = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        total_grads += param.grad.numel()
        nan_count = torch.isnan(param.grad).sum().item()
        nan_grads += nan_count
        if nan_count > 0:
            print(f"  [NaN grad] {name}: {nan_count}/{param.grad.numel()}")

print(f"Gradients: {nan_grads}/{total_grads} NaN ({100*nan_grads/total_grads:.2f}% if total_grads > 0 else 0)")

if nan_grads > 0:
    print("\n  ** BUG: NaN in gradients! **")
    sys.exit(1)

# Step optimizer
scaler.step(optimizer)
scaler.update()

# Check weights after step
nan_weights_after = 0
for name, param in model.named_parameters():
    nan_count = torch.isnan(param).sum().item()
    nan_weights_after += nan_count
    if nan_count > 0:
        print(f"  [NaN weight after step] {name}: {nan_count}/{param.numel()}")

print(f"Weights after step: {nan_weights_after}/{total_params} NaN")

if nan_weights_after > 0:
    print("\n  ** BUG: NaN in weights after optimizer step! **")
    sys.exit(1)

# 8. EMA update
print("\n[8] EMA update...")
model.update_target_encoder()

nan_target = 0
for name, param in model.target_encoder.named_parameters():
    nan_count = torch.isnan(param).sum().item()
    nan_target += nan_count
    if nan_count > 0:
        print(f"  [NaN target] {name}: {nan_count}/{param.numel()}")

if nan_target > 0:
    print("\n  ** BUG: NaN in target encoder after EMA update! **")
    sys.exit(1)

# 9. Second iteration
print("\n[9] Second training iteration...")
optimizer.zero_grad()
with autocast():
    outputs2 = model(latents, masks_context, masks_target, compute_loss=True)
    loss2 = outputs2['loss']

print(f"Loss iteration 2: {loss2.item()}")

if torch.isnan(loss2):
    print("\n  ** BUG: NaN appears on second iteration! **")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED - No NaN detected in training loop!")
print("=" * 60)
print("\nThe bug might be in:")
print("1. DataLoader multiprocessing")
print("2. Different mask patterns")
print("3. Accumulation of small errors over many iterations")
print("4. Specific problematic samples in dataset")
