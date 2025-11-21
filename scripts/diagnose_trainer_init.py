"""
Script de diagnostic pour identifier le blocage dans RJEPATrainer.__init__

Ce script ajoute des prints détaillés à chaque étape de l'initialisation
pour identifier exactement où le code bloque.
"""

import sys
import time
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("DIAGNOSTIC TRAINER INIT - STEP BY STEP")
print("=" * 80)
print()

# 1. Imports
print("[1/10] Importing modules...", flush=True)
start = time.time()
from rjepa.config.settings import Settings
from rjepa.jepa.model import ReasoningJEPA
from rjepa.pipeline.train_rjepa import create_rjepa_model, create_dataloaders
import yaml
print(f"  [OK] Done ({time.time() - start:.2f}s)", flush=True)
print()

# 2. Load config
print("[2/10] Loading config...", flush=True)
start = time.time()
config_path = project_root / "configs" / "rjepa" / "train.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
print(f"  [OK] Config loaded", flush=True)
print(f"    - encoder_embed_dim: {config['model']['encoder_embed_dim']}")
print(f"    - depth_encoder: {config['model']['depth_encoder']}")
print(f"    - depth_predictor: {config['model']['depth_predictor']}")
print(f"    - batch_size: {config['training']['batch_size']}")
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 3. Create dataloaders
print("[3/10] Creating dataloaders...", flush=True)
start = time.time()
train_loader, val_loader, masker_config = create_dataloaders(
    train_latents_dir=Path(config["data"]["train_latents_dir"]),
    val_latents_dir=Path(config["data"].get("val_latents_dir")),
    batch_size=config["training"]["batch_size"],
    num_workers=config["training"].get("num_workers", 0),
    masker_config=config.get("masker", {"type": "contiguous"}),
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print(f"  [OK] Dataloaders created", flush=True)
print(f"    - Train samples: {len(train_loader.dataset)}")
print(f"    - Val samples: {len(val_loader.dataset) if val_loader else 0}")
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 4. Create model
print("[4/10] Creating R-JEPA model...", flush=True)
start = time.time()
model = create_rjepa_model(config["model"])
print(f"  [OK] Model created", flush=True)
num_params = sum(p.numel() for p in model.parameters())
print(f"    - Total params: {num_params:,}")
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 5. Move model to device
print("[5/10] Moving model to GPU...", flush=True)
start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"  [OK] Model on {device}", flush=True)
if device == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"    - CUDA memory allocated: {allocated:.2f} GB")
    print(f"    - CUDA memory reserved: {reserved:.2f} GB")
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 6. Create optimizer
print("[6/10] Creating optimizer...", flush=True)
start = time.time()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["training"]["lr"],
    weight_decay=config["training"]["weight_decay"],
    betas=(0.9, 0.95),
)
print(f"  [OK] Optimizer created", flush=True)
print(f"    - LR: {config['training']['lr']}")
print(f"    - Weight decay: {config['training']['weight_decay']}")
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 7. Calculate scheduler params (CRITICAL - peut bloquer ici!)
print("[7/10] Calculating scheduler parameters...", flush=True)
print(f"  WARNING: len(train_loader) peut être lent avec 21,416 samples!", flush=True)
start = time.time()

# Méthode 1: Calculer manuellement (O(1))
dataset_size = len(train_loader.dataset)
batch_size = config["training"]["batch_size"]
num_batches_manual = (dataset_size + batch_size - 1) // batch_size
print(f"    - Dataset size: {dataset_size}")
print(f"    - Batch size: {batch_size}")
print(f"    - Num batches (manual calc): {num_batches_manual}")
print(f"  Done manual calc ({time.time() - start:.2f}s)", flush=True)

# Méthode 2: Utiliser len(dataloader) (peut être lent!)
print(f"  Now calling len(train_loader)...", flush=True)
start_len = time.time()
num_batches_dataloader = len(train_loader)
len_time = time.time() - start_len
print(f"    - Num batches (len(dataloader)): {num_batches_dataloader}")
print(f"  Done len(train_loader) ({len_time:.2f}s)", flush=True)

if len_time > 5.0:
    print(f"  [WARNING]  WARNING: len(train_loader) took {len_time:.2f}s - THIS IS THE BOTTLENECK!")
print(f"  Total ({time.time() - start:.2f}s)", flush=True)
print()

# 8. Create scheduler
print("[8/10] Creating LR scheduler...", flush=True)
start = time.time()
max_epochs = config["training"]["max_epochs"]
warmup_epochs = config["training"]["warmup_epochs"]
total_steps = num_batches_dataloader * max_epochs
warmup_steps = num_batches_dataloader * warmup_epochs

from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine_schedule(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
print(f"  [OK] Scheduler created", flush=True)
print(f"    - Total steps: {total_steps}")
print(f"    - Warmup steps: {warmup_steps}")
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 9. Create AMP scaler
print("[9/10] Creating AMP scaler...", flush=True)
start = time.time()
from torch.cuda.amp import GradScaler
amp_enabled = config["training"].get("amp_enabled", True)
scaler = GradScaler() if amp_enabled else None
print(f"  [OK] Scaler created (AMP {'enabled' if amp_enabled else 'disabled'})", flush=True)
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

# 10. Test forward pass
print("[10/10] Testing forward pass...", flush=True)
start = time.time()
model.eval()
with torch.no_grad():
    batch = next(iter(train_loader))
    H, domain_ids = batch
    H = H.to(device)
    domain_ids = domain_ids.to(device)
    print(f"    - Batch shape: {H.shape}")
    print(f"    - Domain IDs shape: {domain_ids.shape}")

    outputs = model(H, domain_ids=domain_ids, compute_loss=True)
    print(f"    - Loss: {outputs['loss'].item():.4f}")
print(f"  [OK] Forward pass successful", flush=True)
print(f"  Done ({time.time() - start:.2f}s)", flush=True)
print()

print("=" * 80)
print("DIAGNOSTIC COMPLETE - TRAINER INIT WOULD SUCCEED")
print("=" * 80)
