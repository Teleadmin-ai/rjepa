"""
Student LLM FastAPI Server.

Provides REST API for:
1. Text generation with CoT
2. Latent extraction
"""
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

from rjepa.llm.adapter import LLMAdapter
from rjepa.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global LLM adapter (loaded once at startup)
llm_adapter: Optional[LLMAdapter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    global llm_adapter

    logger.info("Loading Student LLM...")
    llm_adapter = LLMAdapter(
        model_name=settings.student_model_name,
        quantization=settings.student_quantization,
        layer_to_extract=settings.student_layer_to_extract,
    )
    logger.info(f"Student LLM loaded: {llm_adapter}")

    yield

    logger.info("Shutting down Student LLM...")
    del llm_adapter


app = FastAPI(
    title="Student LLM API",
    description="R-JEPA Student LLM with latent extraction",
    version="0.1.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════════


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    num_samples: int = 1
    force_structure: bool = True


class GenerateResponse(BaseModel):
    samples: List[dict]
    model_name: str
    layer_extracted: int


class ExtractLatentsRequest(BaseModel):
    text: str
    step_token: str = "Step"
    layer_idx: Optional[int] = None


class ExtractLatentsResponse(BaseModel):
    latents_shape: List[int]  # [num_steps, hidden_size]
    num_steps: int
    hidden_size: int
    step_boundaries: List[tuple]
    # Note: actual latents returned as separate endpoint due to size


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health")
async def health():
    """Health check endpoint."""
    if llm_adapter is None:
        raise HTTPException(status_code=503, detail="LLM not loaded")

    return {
        "status": "ok",
        "model": llm_adapter.model_name,
        "hidden_size": llm_adapter.hidden_size,
        "num_layers": llm_adapter.num_layers,
        "quantization": llm_adapter.quantization,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate Chain-of-Thought with structured steps.

    Returns:
        {
          "samples": [
            {
              "full_text": "...",
              "steps": ["Step 1: ...", "Step 2: ..."],
              "step_boundaries": [(0, 10), (10, 25), ...]
            },
            ...
          ],
          "model_name": "Qwen/Qwen3-8B-Instruct",
          "layer_extracted": -2
        }
    """
    if llm_adapter is None:
        raise HTTPException(status_code=503, detail="LLM not loaded")

    try:
        results = llm_adapter.generate_with_cot(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            num_samples=request.num_samples,
            force_structure=request.force_structure,
        )

        # Convert to JSON-serializable format (remove tensors)
        samples = []
        for result in results:
            samples.append({
                "full_text": result["full_text"],
                "steps": result["steps"],
                "step_boundaries": result["step_boundaries"],
                "num_tokens": result["tokens"].shape[1],
            })

        return GenerateResponse(
            samples=samples,
            model_name=llm_adapter.model_name,
            layer_extracted=llm_adapter.layer_to_extract,
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_latents", response_model=ExtractLatentsResponse)
async def extract_latents(request: ExtractLatentsRequest):
    """
    Extract latents from text.

    Note: This endpoint returns metadata only (shape, boundaries).
    Actual latent tensors are too large for JSON, use /extract_latents_binary instead.

    Returns:
        {
          "latents_shape": [5, 4096],  # 5 steps, 4096 hidden_size
          "num_steps": 5,
          "hidden_size": 4096,
          "step_boundaries": [(0, 10), (10, 25), ...]
        }
    """
    if llm_adapter is None:
        raise HTTPException(status_code=503, detail="LLM not loaded")

    try:
        # First, generate with CoT to get steps
        results = llm_adapter.generate_with_cot(
            prompt=request.text,
            num_samples=1,
            force_structure=True,
        )
        result = results[0]

        # Extract latents
        latents = llm_adapter.extract_latents(
            tokens=result["tokens"],
            step_boundaries=result["step_boundaries"],
            layer_idx=request.layer_idx,
        )

        return ExtractLatentsResponse(
            latents_shape=list(latents.shape),
            num_steps=latents.shape[0],
            hidden_size=latents.shape[1],
            step_boundaries=result["step_boundaries"],
        )

    except Exception as e:
        logger.error(f"Latent extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def model_info():
    """Get detailed model information."""
    if llm_adapter is None:
        raise HTTPException(status_code=503, detail="LLM not loaded")

    return {
        "model_name": llm_adapter.model_name,
        "hidden_size": llm_adapter.hidden_size,
        "num_layers": llm_adapter.num_layers,
        "quantization": llm_adapter.quantization,
        "layer_to_extract": llm_adapter.layer_to_extract,
        "device": str(llm_adapter.device),
        "dtype": llm_adapter.dtype,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main entrypoint (for local dev)
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "rjepa.llm.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload (model loading is expensive)
    )
