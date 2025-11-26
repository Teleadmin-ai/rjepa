"""
HTTP Client for Student LLM Service.

Provides the same interface as LLMAdapter but calls the Student LLM service via HTTP.
This allows the UI Backend to use a remote LLM service instead of loading the model locally.
"""
import logging
from typing import List, Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)


class StudentLLMClient:
    """
    HTTP client for Student LLM service.

    Mimics the LLMAdapter interface but makes HTTP calls to the service.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
    ):
        """
        Initialize client.

        Args:
            base_url: URL of the Student LLM service
            timeout: Request timeout in seconds (generation can take a while)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

        # Cache model info
        self._model_info: Optional[Dict] = None

    def health(self) -> Dict[str, Any]:
        """Check service health."""
        response = self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def is_available(self) -> bool:
        """Check if service is available."""
        try:
            health = self.health()
            return health.get("status") == "ok"
        except Exception as e:
            logger.warning(f"Student LLM service not available: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self._model_info is None:
            response = self._client.get(f"{self.base_url}/model_info")
            response.raise_for_status()
            self._model_info = response.json()
        return self._model_info

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.get_model_info().get("model_name", "unknown")

    @property
    def hidden_size(self) -> int:
        """Get hidden size."""
        return self.get_model_info().get("hidden_size", 4096)

    @property
    def layer_to_extract(self) -> int:
        """Get layer index for extraction."""
        return self.get_model_info().get("layer_to_extract", -2)

    def generate_with_cot(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        num_samples: int = 1,
        force_structure: bool = True,
        step_token: str = "Step",
        with_latents: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate Chain-of-Thought with structured steps.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_samples: Number of samples to generate
            force_structure: Whether to force step structure
            step_token: Token used for step segmentation (default "Step")
            with_latents: If True, also extract and return latents

        Returns:
            List of dicts with full_text, steps, step_boundaries, num_tokens
            If with_latents=True, also includes 'latents' key with tensor-like list
        """
        if with_latents:
            # Use the combined endpoint that returns both text and latents
            response = self._client.post(
                f"{self.base_url}/generate_with_latents",
                json={
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "num_samples": num_samples,
                    "step_token": step_token,
                },
            )
        else:
            response = self._client.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "num_samples": num_samples,
                    "force_structure": force_structure,
                },
            )
        response.raise_for_status()
        data = response.json()

        # Return samples in the expected format
        return data.get("samples", [])

    def extract_latents(
        self,
        text: str,
        step_token: str = "Step",
        layer_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract latents from text.

        Note: This returns metadata only. The actual latent tensors
        are too large for JSON transport.

        Args:
            text: Input text
            step_token: Token marking step boundaries
            layer_idx: Layer index for extraction

        Returns:
            Dict with latents_shape, num_steps, hidden_size, step_boundaries
        """
        response = self._client.post(
            f"{self.base_url}/extract_latents",
            json={
                "text": text,
                "step_token": step_token,
                "layer_idx": layer_idx,
            },
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"StudentLLMClient(base_url={self.base_url!r})"
