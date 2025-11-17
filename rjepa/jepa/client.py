"""
R-JEPA Client (pour interagir avec le service d'inference).

Client HTTP simple pour communiquer avec l'API R-JEPA.
"""
from typing import List, Dict, Any, Optional
import httpx
import torch


class RJEPAClient:
    """
    Client HTTP pour R-JEPA service.

    Usage:
        client = RJEPAClient("http://localhost:8100")
        score = client.score(latents)
        pred = client.predict_masked(latents, mask_indices)
    """

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 30.0):
        """
        Initialize client.

        Args:
            base_url: Base URL of R-JEPA service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            {
                "status": "ok",
                "model_loaded": true,
                "device": "cuda",
                "model_config": {...}
            }
        """
        response = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def score(
        self,
        latents: torch.Tensor,
        domain_id: Optional[int] = None,
        mask_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Score a latent sequence (compute JEPA-loss).

        Args:
            latents: [num_steps, hidden_dim] tensor
            domain_id: Optional domain ID
            mask_ratio: Ratio of steps to mask (default 0.5)

        Returns:
            {
                "jepa_loss": float,
                "num_steps": int,
                "num_masked": int,
                "device": str
            }
        """
        # Convert tensor to list
        if isinstance(latents, torch.Tensor):
            latents = latents.cpu().tolist()

        payload = {
            "latents": latents,
            "domain_id": domain_id,
            "mask_ratio": mask_ratio,
        }

        response = httpx.post(
            f"{self.base_url}/score", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

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
        # Convert tensor to list
        if isinstance(latents, torch.Tensor):
            latents = latents.cpu().tolist()

        payload = {
            "latents": latents,
            "mask_indices": mask_indices,
            "domain_id": domain_id,
        }

        response = httpx.post(
            f"{self.base_url}/predict_masked", json=payload, timeout=self.timeout
        )
        response.raise_for_status()

        result = response.json()
        predicted_latents = torch.tensor(result["predicted_latents"])

        return predicted_latents
