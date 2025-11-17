"""
Logit Guidance Trainer.

Trains the LogitGuidance module to project R-JEPA latents to vocabulary logit biases.

Training procedure:
1. Load frozen R-JEPA (extract predicted latents)
2. Load LLM (extract true next-token logits)
3. For each step: train guidance to minimize KL(true_dist || guided_dist)
4. Guidance learns to bias logits toward tokens that produce good latents

Key: Both R-JEPA and LLM stay FROZEN, only guidance is trained.
"""
import logging
from pathlib import Path
from typing import Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .logit_guidance import LogitGuidance

logger = logging.getLogger(__name__)


class LogitGuidanceTrainer:
    """
    Trainer for LogitGuidance.

    Philosophy: Guidance is trained AFTER both R-JEPA and LLM are frozen.
    It learns to steer generation toward the latent manifold of good reasoning.
    """

    def __init__(
        self,
        guidance: LogitGuidance,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 3e-4,
        device: str = "cuda",
        use_amp: bool = True,
        grad_clip: float = 1.0,
        log_interval: int = 100,
        wandb_enabled: bool = False,
    ):
        """
        Initialize trainer.

        Args:
            guidance: LogitGuidance instance
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
        self.guidance = guidance.to(device)
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
                guidance.parameters(),
                lr=lr,
                betas=(0.9, 0.95),
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Metrics
        self.global_step = 0
        self.epoch = 0

        logger.info(
            f"LogitGuidanceTrainer initialized: "
            f"device={device}, amp={use_amp}, lr={lr}"
        )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train one epoch.

        Returns:
            Dict of metrics (kl_loss, accuracy)
        """
        self.guidance.train()
        total_kl_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch+1} [Train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            # Batch: {
            #   "latent": [B, latent_dim] - predicted latent from R-JEPA
            #   "llm_logits": [B, vocab_size] - LLM logits before next token
            #   "true_token": [B] - actual next token
            # }
            latent = batch["latent"].to(self.device)
            llm_logits = batch["llm_logits"].to(self.device)
            true_token = batch["true_token"].to(self.device)

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Apply guidance
                    guided_logits = self.guidance.apply_guidance(
                        llm_logits=llm_logits,
                        latent=latent,
                    )

                    # Loss: KL divergence between true distribution and guided distribution
                    # True distribution: one-hot on true_token (or softened)
                    # Guided distribution: softmax(guided_logits)

                    # Option 1: Cross-entropy (simpler)
                    # Loss = CE(guided_logits, true_token)
                    loss = nn.functional.cross_entropy(guided_logits, true_token)

                    # Option 2: KL divergence (more principled)
                    # true_dist = F.softmax(llm_logits, dim=-1)
                    # guided_dist = F.softmax(guided_logits, dim=-1)
                    # loss = F.kl_div(guided_dist.log(), true_dist, reduction='batchmean')

            else:
                guided_logits = self.guidance.apply_guidance(
                    llm_logits=llm_logits,
                    latent=latent,
                )
                loss = nn.functional.cross_entropy(guided_logits, true_token)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.guidance.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.guidance.parameters(), self.grad_clip)
                self.optimizer.step()

            # Metrics
            batch_loss = loss.item()
            total_kl_loss += batch_loss * latent.size(0)

            # Accuracy: guided_logits correctly predict true_token?
            pred_tokens = guided_logits.argmax(dim=-1)
            accuracy = (pred_tokens == true_token).float().mean().item()
            total_accuracy += accuracy * latent.size(0)

            total_samples += latent.size(0)

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_kl_loss / total_samples
                avg_accuracy = total_accuracy / total_samples

                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "acc": f"{avg_accuracy:.4f}",
                    }
                )

                if self.wandb_enabled:
                    try:
                        import wandb

                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/accuracy": avg_accuracy,
                                "train/step": self.global_step,
                            }
                        )
                    except ImportError:
                        pass

            self.global_step += 1

        # Epoch metrics
        avg_loss = total_kl_loss / total_samples
        avg_accuracy = total_accuracy / total_samples

        return {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dict of metrics (loss, accuracy)
        """
        if self.val_loader is None:
            return {}

        self.guidance.eval()
        total_kl_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.epoch+1} [Val]",
            leave=False,
        )

        for batch in pbar:
            latent = batch["latent"].to(self.device)
            llm_logits = batch["llm_logits"].to(self.device)
            true_token = batch["true_token"].to(self.device)

            # Forward
            guided_logits = self.guidance.apply_guidance(
                llm_logits=llm_logits,
                latent=latent,
            )

            loss = nn.functional.cross_entropy(guided_logits, true_token)

            # Metrics
            batch_loss = loss.item()
            total_kl_loss += batch_loss * latent.size(0)

            pred_tokens = guided_logits.argmax(dim=-1)
            accuracy = (pred_tokens == true_token).float().mean().item()
            total_accuracy += accuracy * latent.size(0)

            total_samples += latent.size(0)

        # Validation metrics
        avg_loss = total_kl_loss / total_samples
        avg_accuracy = total_accuracy / total_samples

        logger.info(f"Validation: loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}")

        if self.wandb_enabled:
            try:
                import wandb

                wandb.log(
                    {
                        "val/loss": avg_loss,
                        "val/accuracy": avg_accuracy,
                        "epoch": self.epoch,
                    }
                )
            except ImportError:
                pass

        return {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
        }

    def train(
        self,
        num_epochs: int,
        save_dir: Optional[Path] = None,
        save_interval: int = 1,
    ) -> Dict[str, any]:
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
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])

            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_acc={train_metrics['accuracy']:.4f}"
            )

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])

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
            "guidance_state_dict": self.guidance.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.guidance.config.__dict__,
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

        self.guidance.load_state_dict(checkpoint["guidance_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            f"Checkpoint loaded: step={self.global_step}, epoch={self.epoch}"
        )
