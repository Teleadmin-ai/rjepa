"""
Latent Projection Layers for Multi-LLM Support.

Philosophy:
- R-JEPA learns CONCEPTUAL invariants, not LLM-specific artifacts
- These invariants transfer across models of same family (Qwen3-8B/32B/70B)
- Projections align latent spaces when hidden_size differs

Architecture:
- W_in: LLM latents -> R-JEPA space (compression or identity)
- W_out: R-JEPA space -> LLM latents (expansion or identity, for nudge)
"""
import logging
from typing import Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# LLM hidden sizes (common open-source models)
# This is a REFERENCE - system auto-detects from model config
LLM_HIDDEN_SIZES = {
    # Qwen3 family
    "qwen3-8b": 4096,
    "qwen3-14b": 5120,
    "qwen3-32b": 5120,
    "qwen3-70b": 8192,
    "qwen3-110b": 8192,
    # Llama family
    "llama3-8b": 4096,
    "llama3-70b": 8192,
    "llama3.1-8b": 4096,
    "llama3.1-70b": 8192,
    # Mistral family
    "mistral-7b": 4096,
    "mixtral-8x7b": 4096,
    "mixtral-8x22b": 6144,
    # DeepSeek family
    "deepseek-7b": 4096,
    "deepseek-67b": 8192,
    # Phi family
    "phi-3-mini": 3072,
    "phi-3-medium": 5120,
    # Yi family
    "yi-6b": 4096,
    "yi-34b": 7168,
}


class LatentProjector(nn.Module):
    """
    Linear projection for aligning latent spaces across LLMs.

    Uses orthogonal initialization to preserve norms and distances.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        init_method: str = "orthogonal",
    ):
        """
        Initialize latent projector.

        Args:
            in_dim: Input dimension (source LLM hidden size)
            out_dim: Output dimension (target dimension)
            bias: Whether to use bias (default False for orthogonal)
            init_method: Initialization method ("orthogonal", "xavier", "identity")
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        if in_dim == out_dim:
            # Identity projection (no-op)
            self.proj = nn.Identity()
            self.is_identity = True
        else:
            # Linear projection
            self.proj = nn.Linear(in_dim, out_dim, bias=bias)
            self.is_identity = False

            # Initialize weights
            if init_method == "orthogonal":
                nn.init.orthogonal_(self.proj.weight)
            elif init_method == "xavier":
                nn.init.xavier_uniform_(self.proj.weight)
            elif init_method == "identity":
                # Initialize close to identity (for square matrices)
                if in_dim == out_dim:
                    nn.init.eye_(self.proj.weight)
                else:
                    # For non-square, initialize first min(in,out) rows/cols as identity
                    min_dim = min(in_dim, out_dim)
                    with torch.no_grad():
                        self.proj.weight[:min_dim, :min_dim] = torch.eye(min_dim)

        logger.info(
            f"LatentProjector: {in_dim} -> {out_dim}, "
            f"identity={self.is_identity}, init={init_method}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project latents.

        Args:
            x: [..., in_dim] input latents

        Returns:
            [..., out_dim] projected latents
        """
        return self.proj(x)


class MultiLLMAdapter(nn.Module):
    """
    Adapter for R-JEPA to work with different LLMs.

    Contains:
    - W_in: LLM latents -> R-JEPA space
    - W_out: R-JEPA space -> LLM latents (optional, for nudge)
    - Metadata: LLM info, calibration status
    """

    def __init__(
        self,
        llm_hidden_size: int,
        rjepa_hidden_size: int,
        llm_tag: str = "unknown",
        bidirectional: bool = True,
    ):
        """
        Initialize multi-LLM adapter.

        Args:
            llm_hidden_size: Hidden size of target LLM
            rjepa_hidden_size: Hidden size of R-JEPA (typically from base model)
            llm_tag: LLM identifier (e.g., "qwen3-32b")
            bidirectional: Create both W_in and W_out (True for nudge support)
        """
        super().__init__()

        self.llm_hidden_size = llm_hidden_size
        self.rjepa_hidden_size = rjepa_hidden_size
        self.llm_tag = llm_tag
        self.bidirectional = bidirectional

        # W_in: LLM -> R-JEPA (always needed)
        self.w_in = LatentProjector(
            in_dim=llm_hidden_size,
            out_dim=rjepa_hidden_size,
            init_method="orthogonal",
        )

        # W_out: R-JEPA -> LLM (optional, for nudge mode)
        if bidirectional:
            self.w_out = LatentProjector(
                in_dim=rjepa_hidden_size,
                out_dim=llm_hidden_size,
                init_method="orthogonal",
            )
        else:
            self.w_out = None

        # Calibration status (updated during calibration)
        self.is_calibrated = False
        self.calibration_loss = None

        logger.info(
            f"MultiLLMAdapter: {llm_tag} ({llm_hidden_size}) <-> R-JEPA ({rjepa_hidden_size}), "
            f"bidirectional={bidirectional}"
        )

    def to_rjepa_space(self, llm_latents: torch.Tensor) -> torch.Tensor:
        """
        Project LLM latents to R-JEPA space (W_in).

        Args:
            llm_latents: [..., llm_hidden_size]

        Returns:
            [..., rjepa_hidden_size]
        """
        return self.w_in(llm_latents)

    def to_llm_space(self, rjepa_latents: torch.Tensor) -> torch.Tensor:
        """
        Project R-JEPA latents to LLM space (W_out).

        Args:
            rjepa_latents: [..., rjepa_hidden_size]

        Returns:
            [..., llm_hidden_size]
        """
        if self.w_out is None:
            raise ValueError(
                "W_out not available (bidirectional=False). "
                "Cannot project to LLM space."
            )

        return self.w_out(rjepa_latents)

    def mark_calibrated(self, calibration_loss: float):
        """
        Mark adapter as calibrated.

        Args:
            calibration_loss: Final calibration loss
        """
        self.is_calibrated = True
        self.calibration_loss = calibration_loss

        logger.info(
            f"Adapter {self.llm_tag} marked as calibrated (loss={calibration_loss:.4f})"
        )


def create_adapter_for_llm(
    llm_tag: str,
    llm_hidden_size: Optional[int] = None,
    rjepa_hidden_size: int = 4096,
    bidirectional: bool = True,
    model_config: Optional[dict] = None,
) -> MultiLLMAdapter:
    """
    Factory function to create adapter for ANY open-source LLM.

    Args:
        llm_tag: LLM identifier (e.g., "qwen3-32b", "llama3-70b", "mistral-7b")
        llm_hidden_size: LLM hidden size (auto-detected if None)
        rjepa_hidden_size: R-JEPA hidden size (default 4096 for base model)
        bidirectional: Create W_out for nudge support
        model_config: HuggingFace model config (for auto-detection)

    Returns:
        MultiLLMAdapter instance
    """
    # Auto-detect LLM hidden size if not provided
    if llm_hidden_size is None:
        # Try reference dict first
        llm_hidden_size = LLM_HIDDEN_SIZES.get(llm_tag)

        # If not found, try model config
        if llm_hidden_size is None and model_config is not None:
            llm_hidden_size = model_config.get("hidden_size")

        # Still not found -> error
        if llm_hidden_size is None:
            raise ValueError(
                f"Cannot determine hidden_size for LLM '{llm_tag}'. "
                f"Please provide llm_hidden_size or model_config. "
                f"\nSupported (reference): {list(LLM_HIDDEN_SIZES.keys())}"
            )

        logger.info(
            f"Auto-detected hidden_size={llm_hidden_size} for {llm_tag}"
        )

    return MultiLLMAdapter(
        llm_hidden_size=llm_hidden_size,
        rjepa_hidden_size=rjepa_hidden_size,
        llm_tag=llm_tag,
        bidirectional=bidirectional,
    )


def load_adapter(checkpoint_path: Path) -> MultiLLMAdapter:
    """
    Load adapter from checkpoint.

    Args:
        checkpoint_path: Path to adapter checkpoint (.pth)

    Returns:
        MultiLLMAdapter with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    adapter = MultiLLMAdapter(
        llm_hidden_size=checkpoint["llm_hidden_size"],
        rjepa_hidden_size=checkpoint["rjepa_hidden_size"],
        llm_tag=checkpoint["llm_tag"],
        bidirectional=checkpoint.get("bidirectional", True),
    )

    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.is_calibrated = checkpoint.get("is_calibrated", False)
    adapter.calibration_loss = checkpoint.get("calibration_loss")

    logger.info(f"Loaded adapter from {checkpoint_path}")

    return adapter


def save_adapter(
    adapter: MultiLLMAdapter,
    checkpoint_path: Path,
    metadata: Optional[dict] = None,
):
    """
    Save adapter to checkpoint.

    Args:
        adapter: MultiLLMAdapter to save
        checkpoint_path: Output path (.pth)
        metadata: Additional metadata to save
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "adapter_state_dict": adapter.state_dict(),
        "llm_hidden_size": adapter.llm_hidden_size,
        "rjepa_hidden_size": adapter.rjepa_hidden_size,
        "llm_tag": adapter.llm_tag,
        "bidirectional": adapter.bidirectional,
        "is_calibrated": adapter.is_calibrated,
        "calibration_loss": adapter.calibration_loss,
    }

    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, checkpoint_path)

    logger.info(f"Saved adapter to {checkpoint_path}")


class AdapterTrainer:
    """
    Trainer for calibrating projection weights (W_in/W_out).

    Strategy:
    1. Freeze R-JEPA completely
    2. Train only W_in on small calibration set (5-10% of data)
    3. Minimize R-JEPA reconstruction loss on calibration data
    4. Optionally fine-tune W_out for nudge mode
    """

    def __init__(
        self,
        adapter: MultiLLMAdapter,
        rjepa_model: nn.Module,
        device: str = "cuda",
        lr: float = 1e-4,
    ):
        """
        Initialize adapter trainer.

        Args:
            adapter: MultiLLMAdapter to calibrate
            rjepa_model: Frozen R-JEPA model
            device: Device ("cuda" or "cpu")
            lr: Learning rate
        """
        self.adapter = adapter.to(device)
        self.rjepa_model = rjepa_model.to(device)
        self.device = device

        # Freeze R-JEPA
        for param in self.rjepa_model.parameters():
            param.requires_grad = False

        # Optimizer for adapter only
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr)

        logger.info(
            f"AdapterTrainer initialized: lr={lr}, device={device}, "
            f"adapter_params={sum(p.numel() for p in adapter.parameters() if p.requires_grad)}"
        )

    def train_step(
        self,
        llm_latents: torch.Tensor,
        target_latents: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """
        Single training step for adapter calibration.

        Args:
            llm_latents: [B, S, llm_hidden] latents from new LLM
            target_latents: [B, S, llm_hidden] target latents (from same LLM)
            mask: [B, S] mask for target positions

        Returns:
            Loss value (scalar)
        """
        self.optimizer.zero_grad()

        # Project to R-JEPA space
        rjepa_latents = self.adapter.to_rjepa_space(llm_latents)

        # Forward through R-JEPA (frozen)
        with torch.no_grad():
            rjepa_outputs = self.rjepa_model(
                rjepa_latents,
                compute_loss=False,  # Just get predictions
            )

        # Get predictions in R-JEPA space
        rjepa_pred = rjepa_outputs["pred_masked"]

        # Project targets to R-JEPA space
        rjepa_target = self.adapter.to_rjepa_space(target_latents)

        # Loss: L1 reconstruction in R-JEPA space
        loss = torch.nn.functional.l1_loss(
            rjepa_pred[mask],
            rjepa_target[mask],
        )

        # Backward
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def calibrate(
        self,
        calibration_loader: torch.utils.data.DataLoader,
        num_epochs: int = 3,
    ) -> float:
        """
        Calibrate adapter on calibration set.

        Args:
            calibration_loader: DataLoader with (llm_latents, target_latents, mask)
            num_epochs: Number of calibration epochs

        Returns:
            Final calibration loss
        """
        logger.info(f"Starting calibration for {num_epochs} epochs...")

        self.adapter.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in calibration_loader:
                llm_latents = batch["llm_latents"].to(self.device)
                target_latents = batch["target_latents"].to(self.device)
                mask = batch["mask"].to(self.device)

                loss = self.train_step(llm_latents, target_latents, mask)

                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches

            logger.info(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

        # Mark as calibrated
        self.adapter.mark_calibrated(avg_loss)

        logger.info(f"Calibration complete: final_loss={avg_loss:.4f}")

        return avg_loss
