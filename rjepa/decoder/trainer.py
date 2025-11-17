"""
Latent Decoder Trainer.

Trains the decoder to generate text from R-JEPA latents.

Training procedure:
1. Load frozen R-JEPA (extract latents only)
2. Load validated CoTs (text steps)
3. For each step: extract latent, train decoder to generate text
4. Decoder learns latent→text mapping

Key: R-JEPA stays FROZEN (world model as ground truth).
"""
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm

from .latent_decoder import LatentDecoder

logger = logging.getLogger(__name__)


class LatentDecoderTrainer:
    """
    Trainer for LatentDecoder.

    Philosophy: Decoder is trained AFTER R-JEPA is frozen.
    R-JEPA provides the conceptual latent space (world model),
    decoder learns to verbalize concepts.
    """

    def __init__(
        self,
        model: LatentDecoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-4,
        device: str = "cuda",
        use_amp: bool = True,
        grad_clip: float = 1.0,
        log_interval: int = 100,
        wandb_enabled: bool = False,
    ):
        """
        Initialize trainer.

        Args:
            model: LatentDecoder instance
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (if None, creates AdamW)
            lr: Learning rate
            device: Device ("cuda" or "cpu")
            use_amp: Use automatic mixed precision
            grad_clip: Gradient clipping norm
            log_interval: Log every N batches
            wandb_enabled: Enable W&B logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.wandb_enabled = wandb_enabled

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
        else:
            self.optimizer = optimizer

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Metrics
        self.global_step = 0
        self.epoch = 0

        logger.info(
            f"LatentDecoderTrainer initialized: "
            f"device={device}, amp={use_amp}, lr={lr}"
        )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train one epoch.

        Returns:
            Dict of metrics (loss, perplexity)
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch+1} [Train]",
            leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            latent = batch["latent"].to(self.device)  # [B, latent_dim]
            input_ids = batch["input_ids"].to(self.device)  # [B, S]
            labels = batch["labels"].to(self.device)  # [B, S]

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(latent, input_ids, labels)
                    loss = outputs["loss"]
            else:
                outputs = self.model(latent, input_ids, labels)
                loss = outputs["loss"]

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # Metrics
            batch_loss = loss.item()
            num_tokens = (labels != -100).sum().item()
            total_loss += batch_loss * num_tokens
            total_tokens += num_tokens

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / total_tokens
                perplexity = torch.exp(torch.tensor(avg_loss)).item()

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "ppl": f"{perplexity:.2f}"
                })

                if self.wandb_enabled:
                    try:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/perplexity": perplexity,
                            "train/step": self.global_step,
                        })
                    except ImportError:
                        pass

            self.global_step += 1

        # Epoch metrics
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dict of metrics (loss, perplexity)
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.epoch+1} [Val]",
            leave=False
        )

        for batch in pbar:
            latent = batch["latent"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(latent, input_ids, labels)
            loss = outputs["loss"]

            # Metrics
            batch_loss = loss.item()
            num_tokens = (labels != -100).sum().item()
            total_loss += batch_loss * num_tokens
            total_tokens += num_tokens

        # Validation metrics
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        logger.info(
            f"Validation: loss={avg_loss:.4f}, perplexity={perplexity:.2f}"
        )

        if self.wandb_enabled:
            try:
                import wandb
                wandb.log({
                    "val/loss": avg_loss,
                    "val/perplexity": perplexity,
                    "epoch": self.epoch,
                })
            except ImportError:
                pass

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }

    def train(
        self,
        num_epochs: int,
        save_dir: Optional[Path] = None,
        save_interval: int = 1,
    ) -> Dict[str, Any]:
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            save_interval: Save every N epochs

        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        history = {
            "train_loss": [],
            "train_perplexity": [],
            "val_loss": [],
            "val_perplexity": [],
        }

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])
            history["train_perplexity"].append(train_metrics["perplexity"])

            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_ppl={train_metrics['perplexity']:.2f}"
            )

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                history["val_loss"].append(val_metrics["loss"])
                history["val_perplexity"].append(val_metrics["perplexity"])

            # Save checkpoint
            if save_dir and (epoch + 1) % save_interval == 0:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_path = save_dir / f"checkpoint-epoch-{epoch+1}.pth"
                self.save_checkpoint(checkpoint_path)

                logger.info(f"Checkpoint saved: {checkpoint_path}")

        logger.info("Training completed!")
        return history

    def save_checkpoint(self, path: Path):
        """
        Save checkpoint.

        Args:
            path: Checkpoint path
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config.__dict__,
            "global_step": self.global_step,
            "epoch": self.epoch,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """
        Load checkpoint.

        Args:
            path: Checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            f"Checkpoint loaded: step={self.global_step}, epoch={self.epoch}"
        )
