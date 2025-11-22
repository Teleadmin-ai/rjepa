"""
R-JEPA Trainer.

Handles training loop with:
- EMA target encoder updates
- Gradient clipping
- AMP (Automatic Mixed Precision)
- Checkpointing
- W&B logging
- LR scheduling
"""
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm
import sys
import time

logger = logging.getLogger(__name__)


class RJEPATrainer:
    """
    Trainer for R-JEPA model.

    Philosophy: This trains a world model of reasoning. The goal is not
    to predict tokens, but to learn the stable relationships between
    reasoning steps in latent space.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        masker_config: Optional[Dict] = None,  # OPTION 2: Config for GPU-based masking
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        grad_clip: float = 1.0,
        amp_enabled: bool = True,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 0.9999,
        checkpoint_dir: Optional[Path] = None,
        log_interval: int = 10,
        val_interval: int = 1,
        use_wandb: bool = False,
        wandb_config: Optional[Dict] = None,
        device: str = "cuda",
    ):
        """
        Initialize trainer.

        Args:
            model: R-JEPA model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            optimizer: Optional custom optimizer (if None, creates AdamW)
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            warmup_epochs: Warmup epochs for LR scheduler
            max_epochs: Maximum training epochs
            grad_clip: Gradient clipping value
            amp_enabled: Enable Automatic Mixed Precision
            ema_momentum_start: Initial EMA momentum
            ema_momentum_end: Final EMA momentum (annealed)
            checkpoint_dir: Directory to save checkpoints
            log_interval: Log every N steps
            val_interval: Validate every N epochs
            use_wandb: Enable W&B logging
            wandb_config: W&B configuration dict
            device: Device for training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # OPTION 2: Store masker config for GPU-based masking
        self.masker_config = masker_config if masker_config is not None else {"type": "contiguous"}
        self.masker = None  # Will be created in train_epoch() on GPU

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
        else:
            self.optimizer = optimizer

        # Scheduler (warmup + cosine decay)
        total_steps = len(train_loader) * max_epochs
        warmup_steps = len(train_loader) * warmup_epochs

        self.scheduler = self._create_scheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        # AMP
        self.amp_enabled = amp_enabled
        self.scaler = GradScaler() if amp_enabled else None

        # Training config
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.accumulation_steps = 1  # Will be set from config if needed

        # EMA config
        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # W&B
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                if not wandb.run:
                    wandb.init(**(wandb_config or {}))
                logger.info("W&B logging enabled")
            except ImportError:
                logger.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        logger.info(
            f"RJEPATrainer initialized: "
            f"lr={lr}, epochs={max_epochs}, warmup={warmup_epochs}, "
            f"amp={amp_enabled}, device={device}"
        )

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
    ):
        """Create LR scheduler with warmup + cosine decay."""
        from torch.optim.lr_scheduler import LambdaLR
        import math

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    def _get_ema_momentum(self) -> float:
        """Get current EMA momentum (annealed from start to end)."""
        progress = self.current_epoch / self.max_epochs
        return (
            self.ema_momentum_start
            + (self.ema_momentum_end - self.ema_momentum_start) * progress
        )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train one epoch.

        Returns:
            Dict with epoch metrics
        """
        self.model.train()

        epoch_losses = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "var_reg_loss": 0.0,
        }
        num_batches = 0

        # OPTION 2: Create masker once (lazy init) for GPU-based masking
        if self.masker is None:
            from rjepa.jepa.maskers import create_masker, MaskCollator
            masker = create_masker(self.masker_config)
            self.masker = MaskCollator(masker, device=self.device)
            logger.info(f"Created masker on {self.device}: {self.masker_config}")

        # Use tqdm with file=sys.stderr and force flush for log compatibility
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            total=len(self.train_loader),
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=1.0,  # Update every second
        )
        for batch_idx, batch in enumerate(pbar):
            # OPTION 2: DataLoader returns list of (H, domain_id) tuples
            # batch = [(H1, domain_id1), (H2, domain_id2), ...]
            # where each H has shape [num_steps, hidden_size]

            # OPTION 2: Generate masks on GPU (dynamic masking for generalization)
            # This is the CRITICAL part - masking happens HERE, not in DataLoader workers
            # Overhead: ~0.5ms per batch (0.4% of 115ms forward pass)
            # MaskCollator will:
            # 1. Move tensors to GPU
            # 2. Generate masks dynamically
            # 3. Return dict with keys: latents, context_mask, target_mask
            masked_batch = self.masker(batch)

            # Extract masks from collated batch (all on CPU from MaskCollator)
            context_mask_bool = masked_batch["context_mask"]
            target_mask_bool = masked_batch["target_mask"]
            latents = masked_batch["latents"]
            domain_ids = masked_batch.get("domain_ids")

            # Move tensors to GPU in main thread (workers can't create CUDA tensors on Windows)
            latents = latents.to(self.device)
            context_mask_bool = context_mask_bool.to(self.device)
            target_mask_bool = target_mask_bool.to(self.device)
            if domain_ids is not None:
                domain_ids = domain_ids.to(self.device)

            # Convert boolean masks to index masks for V-JEPA format
            # V-JEPA expects: List of [B, K] tensors where K = number of positions to keep
            B = latents.size(0)
            context_mask = []
            target_mask = []
            for b in range(B):
                ctx_idx = (context_mask_bool[b]).nonzero(as_tuple=False).squeeze(-1)  # [K_c]
                tgt_idx = (target_mask_bool[b]).nonzero(as_tuple=False).squeeze(-1)  # [K_t]
                context_mask.append(ctx_idx.unsqueeze(0))  # [1, K_c]
                target_mask.append(tgt_idx.unsqueeze(0))  # [1, K_t]

            # Stack into [B, K] tensors (pad if needed)
            max_ctx = max(m.size(1) for m in context_mask)
            max_tgt = max(m.size(1) for m in target_mask)

            context_mask_tensor = torch.zeros(B, max_ctx, dtype=torch.long, device=self.device)
            target_mask_tensor = torch.zeros(B, max_tgt, dtype=torch.long, device=self.device)

            for b in range(B):
                ctx_len = context_mask[b].size(1)
                tgt_len = target_mask[b].size(1)
                context_mask_tensor[b, :ctx_len] = context_mask[b]
                target_mask_tensor[b, :tgt_len] = target_mask[b]

            # Wrap in list for V-JEPA format (single mask per batch)
            context_mask = [context_mask_tensor]
            target_mask = [target_mask_tensor]

            # Forward pass with AMP
            # Only zero grad at start of accumulation
            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.zero_grad()

            if self.amp_enabled:
                with autocast():
                    outputs = self.model(
                        latents,
                        context_mask,
                        target_mask,
                        domain_ids,
                        compute_loss=True,
                    )
                    loss = outputs["loss"]
                    # Scale loss by accumulation steps
                    loss = loss / self.accumulation_steps

                # FIX: Direct backward WITHOUT GradScaler (GradScaler causes 99.9% NaN gradients!)
                # AMP autocast still works for memory/speed, but we skip gradient scaling
                loss.backward()

                # Only step optimizer every accumulation_steps batches
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping (direct, no scaler.unscale_ needed)
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )

                    # Optimizer step (direct, no scaler.step needed)
                    self.optimizer.step()

                    # LR scheduler step
                    self.scheduler.step()

                    # EMA update with annealed momentum
                    self.model.ema_momentum = self._get_ema_momentum()
                    self.model.update_target_encoder()
            else:
                outputs = self.model(
                    latents,
                    context_mask,
                    target_mask,
                    domain_ids,
                    compute_loss=True,
                )
                loss = outputs["loss"]
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps

                # Backward
                loss.backward()

                # Only step optimizer every accumulation_steps batches
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )

                    # Optimizer step
                    self.optimizer.step()

                    # LR scheduler step
                    self.scheduler.step()

                    # EMA update with annealed momentum
                    self.model.ema_momentum = self._get_ema_momentum()
                    self.model.update_target_encoder()

            # Accumulate losses
            epoch_losses["loss"] += loss.item()
            epoch_losses["recon_loss"] += outputs["recon_loss"].item()
            epoch_losses["var_reg_loss"] += outputs["var_reg_loss"].item()
            num_batches += 1

            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})

            # Log to W&B
            if self.use_wandb and self.global_step % self.log_interval == 0:
                self.wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/recon_loss": outputs["recon_loss"].item(),
                        "train/var_reg_loss": outputs["var_reg_loss"].item(),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/ema_momentum": self.model.ema_momentum,
                        "epoch": self.current_epoch,
                        "step": self.global_step,
                    }
                )

            self.global_step += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dict with validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        val_losses = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "var_reg_loss": 0.0,
        }
        num_batches = 0

        from rich.progress import track

        # OPTION 2: Create masker once for GPU-based masking (same as train_epoch)
        if self.masker is None:
            from rjepa.jepa.maskers import create_masker, MaskCollator
            masker = create_masker(self.masker_config)
            self.masker = MaskCollator(masker, device=self.device)

        for batch in track(self.val_loader, description="Validation", total=len(self.val_loader)):
            # OPTION 2: Apply masks on GPU (batch is list of tuples from simple_collate)
            masked_batch = self.masker(batch)

            # Extract tensors and move to GPU
            latents = masked_batch["latents"].to(self.device)
            context_mask = masked_batch["context_mask"].to(self.device)
            target_mask = masked_batch["target_mask"].to(self.device)
            domain_ids = masked_batch.get("domain_ids")
            if domain_ids is not None:
                domain_ids = domain_ids.to(self.device)

            # Forward pass
            if self.amp_enabled:
                with autocast():
                    outputs = self.model(
                        latents,
                        context_mask,
                        target_mask,
                        domain_ids,
                        compute_loss=True,
                    )
            else:
                outputs = self.model(
                    latents,
                    context_mask,
                    target_mask,
                    domain_ids,
                    compute_loss=True,
                )

            # Accumulate losses
            val_losses["loss"] += outputs["loss"].item()
            val_losses["recon_loss"] += outputs["recon_loss"].item()
            val_losses["var_reg_loss"] += outputs["var_reg_loss"].item()
            num_batches += 1

        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def save_checkpoint(
        self,
        filename: str = "checkpoint.pth",
        is_best: bool = False,
    ):
        """
        Save checkpoint.

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(
            f"Resumed from epoch {self.current_epoch}, "
            f"step {self.global_step}, "
            f"best_val_loss={self.best_val_loss:.4f}"
        )

    def train(self):
        """
        Main training loop.

        Returns:
            Training history dict
        """
        logger.info("=" * 80)
        logger.info("STARTING R-JEPA TRAINING")
        logger.info("=" * 80)

        start_time = time.time()

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_metrics = self.train_epoch()

            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"recon_loss={train_metrics['recon_loss']:.4f}, "
                f"var_reg={train_metrics['var_reg_loss']:.4f}"
            )

            # Validation
            if self.val_loader is not None and epoch % self.val_interval == 0:
                val_metrics = self.validate()

                logger.info(
                    f"Validation: "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"recon_loss={val_metrics['recon_loss']:.4f}"
                )

                # Log to W&B
                if self.use_wandb:
                    self.wandb.log(
                        {
                            "val/loss": val_metrics["loss"],
                            "val/recon_loss": val_metrics["recon_loss"],
                            "val/var_reg_loss": val_metrics["var_reg_loss"],
                            "epoch": epoch,
                        }
                    )

                # Save best model
                is_best = val_metrics["loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["loss"]
                    logger.info(
                        f"New best validation loss: {self.best_val_loss:.4f}"
                    )

                # Save checkpoint
                self.save_checkpoint(
                    filename=f"checkpoint-epoch-{epoch}.pth",
                    is_best=is_best,
                )
            else:
                # Save periodic checkpoint
                if epoch % 10 == 0:
                    self.save_checkpoint(filename=f"checkpoint-epoch-{epoch}.pth")

        elapsed = time.time() - start_time

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {elapsed / 3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)

        # Save final checkpoint
        self.save_checkpoint(filename="final.pth")

        return {
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.max_epochs,
            "total_time": elapsed,
        }
