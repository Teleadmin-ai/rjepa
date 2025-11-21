"""
R-JEPA Inference Service (FastAPI).

Expose R-JEPA model via REST API pour:
- Scoring (JEPA-loss) pour re-ranking
- Prediction de steps masqués pour nudge/plan
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rjepa.jepa.model import ReasoningJEPA, create_rjepa_model

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════


class LatentSequenceRequest(BaseModel):
    """Request schema for latent sequence."""

    latents: List[List[float]] = Field(
        ...,
        description="Latent sequence [num_steps, hidden_dim]",
    )
    domain_id: Optional[int] = Field(
        None,
        description="Domain ID (optional)",
    )


class ScoreRequest(LatentSequenceRequest):
    """Request schema for scoring."""

    mask_ratio: float = Field(
        0.5,
        description="Ratio of steps to mask for scoring (default 0.5)",
        ge=0.1,
        le=0.9,
    )


class ScoreResponse(BaseModel):
    """Response schema for scoring."""

    jepa_loss: float = Field(..., description="JEPA loss (lower = better coherence)")
    num_steps: int = Field(..., description="Number of steps in sequence")
    num_masked: int = Field(..., description="Number of masked steps")
    device: str = Field(..., description="Device used for inference")


class PredictMaskedRequest(LatentSequenceRequest):
    """Request schema for predicting masked steps."""

    mask_indices: List[int] = Field(
        ...,
        description="Indices of steps to mask and predict",
    )


class PredictMaskedResponse(BaseModel):
    """Response schema for predicting masked steps."""

    predicted_latents: List[List[float]] = Field(
        ...,
        description="Predicted latents for masked steps [num_masked, hidden_dim]",
    )
    num_predicted: int = Field(..., description="Number of predicted steps")
    device: str = Field(..., description="Device used for inference")


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device model is on")
    rjepa_config: Dict[str, Any] = Field(..., description="R-JEPA model configuration")


# ═════════════════════════════════════════════════════════════════════════════
# R-JEPA SERVICE
# ═════════════════════════════════════════════════════════════════════════════


class RJEPAService:
    """
    R-JEPA inference service.

    Loads a trained R-JEPA checkpoint and exposes inference endpoints.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize service.

        Args:
            checkpoint_path: Path to R-JEPA checkpoint (.pth)
            device: Device to run inference on
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model: Optional[ReasoningJEPA] = None
        self.model_config: Optional[Dict] = None

        # Load model
        self._load_model()

    def _load_model(self):
        """Load R-JEPA model from checkpoint."""
        logger.info(f"Loading R-JEPA checkpoint from {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract model config
        self.model_config = checkpoint.get("model_config", {})

        if not self.model_config:
            raise ValueError(
                "Checkpoint does not contain model_config. "
                "Please provide a valid R-JEPA checkpoint."
            )

        # Create model from config
        self.model = create_rjepa_model(self.model_config)

        # Load weights (strict=False to allow domain_embed mismatch)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            raise ValueError(
                "Checkpoint does not contain model_state_dict. "
                "Please provide a valid R-JEPA checkpoint."
            )

        # Move to device and eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Model config: {self.model_config}")

    @torch.no_grad()
    def score(
        self,
        latents: torch.Tensor,
        domain_id: Optional[int] = None,
        mask_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute JEPA-loss for a latent sequence (for re-ranking).

        Args:
            latents: [num_steps, hidden_dim] tensor
            domain_id: Optional domain ID
            mask_ratio: Ratio of steps to mask (default 0.5)

        Returns:
            {
                "jepa_loss": float,
                "num_steps": int,
                "num_masked": int,
            }
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Add batch dimension
        latents = latents.unsqueeze(0).to(self.device)  # [1, S, D]

        # Create context and target masks
        num_steps = latents.shape[1]
        num_masked = int(num_steps * mask_ratio)

        # Use contiguous masking (mask middle block)
        start_idx = (num_steps - num_masked) // 2
        context_mask = torch.ones(1, num_steps, dtype=torch.bool, device=self.device)
        context_mask[0, start_idx : start_idx + num_masked] = False

        target_mask = ~context_mask

        # Prepare domain_ids
        domain_ids = None
        if domain_id is not None:
            domain_ids = torch.tensor([domain_id], device=self.device)

        # Forward pass
        outputs = self.model(
            latents,
            context_mask=context_mask,
            target_mask=target_mask,
            domain_ids=domain_ids,
            compute_loss=True,
        )

        return {
            "jepa_loss": outputs["loss"].item(),
            "num_steps": num_steps,
            "num_masked": num_masked,
        }

    @torch.no_grad()
    def predict_masked(
        self,
        latents: torch.Tensor,
        mask_indices: List[int],
        domain_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Predict latents for masked steps.

        Args:
            latents: [num_steps, hidden_dim] tensor
            mask_indices: Indices of steps to mask and predict
            domain_id: Optional domain ID

        Returns:
            predicted_latents: [num_masked, hidden_dim] tensor
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Add batch dimension
        latents = latents.unsqueeze(0).to(self.device)  # [1, S, D]

        # Create masks
        num_steps = latents.shape[1]
        context_mask = torch.ones(1, num_steps, dtype=torch.bool, device=self.device)
        context_mask[0, mask_indices] = False

        target_mask = ~context_mask

        # Prepare domain_ids
        domain_ids = None
        if domain_id is not None:
            domain_ids = torch.tensor([domain_id], device=self.device)

        # Forward pass
        outputs = self.model(
            latents,
            context_mask=context_mask,
            target_mask=target_mask,
            domain_ids=domain_ids,
            compute_loss=False,
        )

        # Extract predictions for masked positions
        pred = outputs["pred"][0]  # [S, hidden_dim]
        pred_masked = pred[mask_indices]  # [num_masked, hidden_dim]

        return pred_masked.cpu()


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═════════════════════════════════════════════════════════════════════════════

# Global service instance
service: Optional[RJEPAService] = None


def create_app(
    checkpoint_path: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> FastAPI:
    """
    Create FastAPI app with R-JEPA service.

    Args:
        checkpoint_path: Path to R-JEPA checkpoint
        device: Device to run inference on

    Returns:
        FastAPI app
    """
    global service

    app = FastAPI(
        title="R-JEPA Inference Service",
        description="World model for text reasoning in latent space",
        version="0.1.0",
    )

    # Initialize service
    service = RJEPAService(checkpoint_path=checkpoint_path, device=device)

    @app.get("/health", response_model=HealthResponse)
    def health():
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            model_loaded=service.model is not None,
            device=service.device,
            rjepa_config=service.model_config or {},
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest):
        """
        Score a latent sequence (compute JEPA-loss).

        Lower loss = better coherence with learned world model.
        Use this for re-ranking CoT candidates.
        """
        try:
            # Convert to tensor
            latents_tensor = torch.tensor(
                request.latents, dtype=torch.float32
            )  # [S, D]

            # Validate shape
            if latents_tensor.dim() != 2:
                raise ValueError(
                    f"Latents must be 2D [num_steps, hidden_dim], got shape {latents_tensor.shape}"
                )

            # Score
            result = service.score(
                latents=latents_tensor,
                domain_id=request.domain_id,
                mask_ratio=request.mask_ratio,
            )

            return ScoreResponse(
                jepa_loss=result["jepa_loss"],
                num_steps=result["num_steps"],
                num_masked=result["num_masked"],
                device=service.device,
            )

        except Exception as e:
            logger.error(f"Error in /score: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict_masked", response_model=PredictMaskedResponse)
    def predict_masked(request: PredictMaskedRequest):
        """
        Predict latents for masked steps.

        Use this for nudge (correcting steps) or plan (completing missing steps).
        """
        try:
            # Convert to tensor
            latents_tensor = torch.tensor(
                request.latents, dtype=torch.float32
            )  # [S, D]

            # Validate shape
            if latents_tensor.dim() != 2:
                raise ValueError(
                    f"Latents must be 2D [num_steps, hidden_dim], got shape {latents_tensor.shape}"
                )

            # Validate mask_indices
            num_steps = latents_tensor.shape[0]
            if any(idx >= num_steps or idx < 0 for idx in request.mask_indices):
                raise ValueError(
                    f"mask_indices contains invalid indices. "
                    f"Valid range: [0, {num_steps-1}], got {request.mask_indices}"
                )

            # Predict
            pred_latents = service.predict_masked(
                latents=latents_tensor,
                mask_indices=request.mask_indices,
                domain_id=request.domain_id,
            )

            return PredictMaskedResponse(
                predicted_latents=pred_latents.tolist(),
                num_predicted=len(request.mask_indices),
                device=service.device,
            )

        except Exception as e:
            logger.error(f"Error in /predict_masked: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run R-JEPA inference service")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to R-JEPA checkpoint (.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8100,
        help="Port to bind to (default: 8100)",
    )

    args = parser.parse_args()

    # Create app
    app = create_app(checkpoint_path=args.checkpoint, device=args.device)

    # Run server
    logger.info(f"Starting R-JEPA service on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
