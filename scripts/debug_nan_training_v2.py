"""
Debug script v2 - Test backward with and without AMP
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
print("DEBUG v2: Testing backward WITH vs WITHOUT AMP")
print("=" * 60)

# Load config
with open("configs/rjepa/train.yaml") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create fresh model
print("\n[1] Creating fresh model...")
model = create_rjepa_model(config['model'])
model = model.to(device)
model.train()

# Load data
print("\n[2] Loading data...")
dataset = LatentDatasetMultiShard(
    latents_dir=Path(config['data']['train_latents_dir']),
)
latents, domain_id = dataset[0]
latents = latents.unsqueeze(0).to(device)

# Create masks
B, S, D = latents.shape
masker = ContiguousMasker(min_mask_ratio=0.3, max_mask_ratio=0.7)
context_mask_bool, target_mask_bool = masker(B, S)

# Convert to indices
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

print(f"Batch shape: {latents.shape}")
print(f"Context steps: {context_mask_tensor.shape[1]}, Target steps: {target_mask_tensor.shape[1]}")

# =========================================
# TEST 1: Backward WITHOUT AMP (FP32)
# =========================================
print("\n" + "=" * 60)
print("[TEST 1] Backward WITHOUT AMP (full FP32)")
print("=" * 60)

model_fp32 = create_rjepa_model(config['model']).to(device)
model_fp32.train()
optimizer_fp32 = torch.optim.AdamW(model_fp32.parameters(), lr=1e-4)

optimizer_fp32.zero_grad()
outputs_fp32 = model_fp32(latents, masks_context, masks_target, compute_loss=True)
loss_fp32 = outputs_fp32['loss']
print(f"Loss (FP32): {loss_fp32.item()}")

loss_fp32.backward()

nan_grads_fp32 = 0
total_grads_fp32 = 0
for name, param in model_fp32.named_parameters():
    if param.grad is not None:
        total_grads_fp32 += param.grad.numel()
        nan_count = torch.isnan(param.grad).sum().item()
        nan_grads_fp32 += nan_count
        if nan_count > 0:
            print(f"  [NaN grad FP32] {name}: {nan_count}/{param.grad.numel()}")

print(f"FP32 Gradients: {nan_grads_fp32}/{total_grads_fp32} NaN ({100*nan_grads_fp32/total_grads_fp32:.2f}%)")

if nan_grads_fp32 == 0:
    print("  *** FP32 backward is CLEAN - no NaN! ***")
else:
    print("  *** FP32 backward has NaN - problem is NOT AMP ***")

# =========================================
# TEST 2: Backward WITH AMP but NO GradScaler
# =========================================
print("\n" + "=" * 60)
print("[TEST 2] Backward WITH AMP, WITHOUT GradScaler")
print("=" * 60)

model_amp_noscale = create_rjepa_model(config['model']).to(device)
model_amp_noscale.train()
optimizer_amp_noscale = torch.optim.AdamW(model_amp_noscale.parameters(), lr=1e-4)

optimizer_amp_noscale.zero_grad()

# Use new API
with torch.amp.autocast('cuda'):
    outputs_amp_noscale = model_amp_noscale(latents, masks_context, masks_target, compute_loss=True)
    loss_amp_noscale = outputs_amp_noscale['loss']

print(f"Loss (AMP no scale): {loss_amp_noscale.item()}")

# Backward without scaler
loss_amp_noscale.backward()

nan_grads_amp_noscale = 0
total_grads_amp_noscale = 0
for name, param in model_amp_noscale.named_parameters():
    if param.grad is not None:
        total_grads_amp_noscale += param.grad.numel()
        nan_count = torch.isnan(param.grad).sum().item()
        nan_grads_amp_noscale += nan_count
        if nan_count > 0:
            print(f"  [NaN grad AMP no scale] {name}: {nan_count}/{param.grad.numel()}")

print(f"AMP (no scale) Gradients: {nan_grads_amp_noscale}/{total_grads_amp_noscale} NaN ({100*nan_grads_amp_noscale/total_grads_amp_noscale:.2f}%)")

if nan_grads_amp_noscale == 0:
    print("  *** AMP without scaler is CLEAN - problem is GradScaler! ***")
else:
    print("  *** AMP without scaler has NaN - problem is autocast itself ***")

# =========================================
# TEST 3: Backward WITH AMP and GradScaler
# =========================================
print("\n" + "=" * 60)
print("[TEST 3] Backward WITH AMP and GradScaler")
print("=" * 60)

model_amp_scale = create_rjepa_model(config['model']).to(device)
model_amp_scale.train()
optimizer_amp_scale = torch.optim.AdamW(model_amp_scale.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda')

optimizer_amp_scale.zero_grad()

with torch.amp.autocast('cuda'):
    outputs_amp_scale = model_amp_scale(latents, masks_context, masks_target, compute_loss=True)
    loss_amp_scale = outputs_amp_scale['loss']

print(f"Loss (AMP with scale): {loss_amp_scale.item()}")

# Backward WITH scaler
scaler.scale(loss_amp_scale).backward()

nan_grads_amp_scale = 0
total_grads_amp_scale = 0
for name, param in model_amp_scale.named_parameters():
    if param.grad is not None:
        total_grads_amp_scale += param.grad.numel()
        nan_count = torch.isnan(param.grad).sum().item()
        nan_grads_amp_scale += nan_count

print(f"AMP (with scale) Gradients: {nan_grads_amp_scale}/{total_grads_amp_scale} NaN ({100*nan_grads_amp_scale/total_grads_amp_scale:.2f}%)")

if nan_grads_amp_scale > 0:
    print("  *** AMP with GradScaler produces NaN! ***")

# =========================================
# SUMMARY
# =========================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"FP32 (no AMP):       {nan_grads_fp32}/{total_grads_fp32} NaN")
print(f"AMP (no scaler):     {nan_grads_amp_noscale}/{total_grads_amp_noscale} NaN")
print(f"AMP (with scaler):   {nan_grads_amp_scale}/{total_grads_amp_scale} NaN")
print()

if nan_grads_fp32 == 0 and nan_grads_amp_noscale == 0 and nan_grads_amp_scale > 0:
    print("*** CONCLUSION: GradScaler is causing NaN! ***")
    print("*** FIX: Disable GradScaler OR use higher init_scale ***")
elif nan_grads_fp32 == 0 and nan_grads_amp_noscale > 0:
    print("*** CONCLUSION: autocast itself is causing NaN! ***")
    print("*** FIX: Disable AMP entirely ***")
elif nan_grads_fp32 > 0:
    print("*** CONCLUSION: Model has NaN even in FP32! ***")
    print("*** FIX: Check model architecture/loss function ***")
else:
    print("All tests passed - no NaN detected!")
