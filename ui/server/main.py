"""
UI Backend Gateway (FastAPI).

Agrège les appels vers:
- student-llm (génération)
- rjepa-service (scoring/prediction)
- prefect (jobs monitoring)
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rjepa.llm.adapter import LLMAdapter
from rjepa.jepa.client import RJEPAClient
from rjepa.inference import (
    rerank_cots_with_jepa,
    nudge_with_regeneration,
    auto_complete_missing_steps,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    """Request schema for chat."""

    prompt: str = Field(..., description="User question/prompt")
    mode: str = Field(
        "off",
        description="JEPA mode: 'off', 'rerank', 'nudge', 'plan'",
    )
    num_samples: int = Field(
        4,
        description="Number of candidates for rerank mode",
        ge=1,
        le=10,
    )
    temperature: float = Field(
        0.7,
        description="Generation temperature",
        ge=0.0,
        le=2.0,
    )
    domain_id: Optional[int] = Field(
        None,
        description="Optional domain ID",
    )


class ChatResponse(BaseModel):
    """Response schema for chat."""

    answer: str = Field(..., description="Final answer")
    steps: List[str] = Field(default_factory=list, description="Reasoning steps")
    jepa_score: Optional[float] = Field(None, description="JEPA score (if applicable)")
    mode: str = Field(..., description="Mode used")
    candidates: Optional[List[Dict]] = Field(
        None,
        description="All candidates (for rerank mode)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class FeedbackRequest(BaseModel):
    """Request schema for user feedback."""

    session_id: str
    prompt: str
    answer: str
    feedback: str = Field(..., description="'thumbs_up', 'thumbs_down', or comment")
    jepa_score: Optional[float] = None


class JobStatus(BaseModel):
    """Job status schema."""

    job_id: str
    job_type: str
    status: str
    progress: float
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═════════════════════════════════════════════════════════════════════════════


def create_app(
    student_llm_url: str = "http://localhost:8000",
    rjepa_service_url: str = "http://localhost:8100",
    llm_model_name: str = "Qwen/Qwen3-8B-Instruct",
) -> FastAPI:
    """
    Create FastAPI app for UI backend.

    Args:
        student_llm_url: URL of student LLM service
        rjepa_service_url: URL of R-JEPA service
        llm_model_name: Model name for LLM adapter

    Returns:
        FastAPI app
    """
    app = FastAPI(
        title="R-JEPA UI Backend",
        description="Gateway for R-JEPA web interface",
        version="0.1.0",
    )

    # CORS middleware (pour Next.js dev server)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize clients
    # Note: En production, on utiliserait des clients HTTP vers les services
    # Pour le MVP, on instancie directement les adapters
    try:
        llm = LLMAdapter(
            model_name=llm_model_name,
            device="cuda",
            quantization="awq-4bit",
            layer_to_extract=-2,
        )
        logger.info(f"LLM adapter initialized: {llm_model_name}")
    except Exception as e:
        logger.warning(f"Failed to initialize LLM adapter: {e}")
        llm = None

    rjepa_client = RJEPAClient(base_url=rjepa_service_url)

    @app.get("/health")
    def health():
        """Health check."""
        return {
            "status": "ok",
            "llm_loaded": llm is not None,
            "rjepa_url": rjepa_service_url,
        }

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Chat endpoint avec support JEPA modes.

        Modes:
        - 'off': LLM seul (baseline)
        - 'rerank': Génère N candidates, choisit meilleure (JEPA-guided)
        - 'nudge': Correction latente en temps réel
        - 'plan': Complétion de steps manquants
        """
        if llm is None:
            raise HTTPException(
                status_code=503,
                detail="LLM not loaded",
            )

        logger.info(f"Chat request: mode={request.mode}, prompt={request.prompt[:100]}...")

        try:
            if request.mode == "off":
                # Baseline: LLM seul
                result = llm.generate_with_cot(
                    prompt=request.prompt,
                    max_new_tokens=512,
                    temperature=request.temperature,
                    num_samples=1,
                )[0]

                return ChatResponse(
                    answer=result["full_text"],
                    steps=result["steps"],
                    mode="off",
                    metadata={
                        "num_steps": len(result["steps"]),
                    },
                )

            elif request.mode == "rerank":
                # Re-ranking mode
                result = rerank_cots_with_jepa(
                    prompt=request.prompt,
                    llm=llm,
                    rjepa_client=rjepa_client,
                    num_samples=request.num_samples,
                    temperature=request.temperature,
                    domain_id=request.domain_id,
                )

                best_cot = result["best_cot"]

                return ChatResponse(
                    answer=best_cot["full_text"],
                    steps=best_cot["steps"],
                    jepa_score=best_cot["jepa_loss"],
                    mode="rerank",
                    candidates=[
                        {
                            "text": c["full_text"],
                            "score": c["score"],
                            "jepa_loss": c["jepa_loss"],
                        }
                        for c in result["candidates"]
                    ],
                    metadata={
                        "num_candidates": result["num_candidates"],
                        "num_steps": best_cot["num_steps"],
                    },
                )

            elif request.mode == "nudge":
                # Nudge mode (avec regeneration)
                result = nudge_with_regeneration(
                    prompt=request.prompt,
                    llm=llm,
                    rjepa_client=rjepa_client,
                    max_attempts=3,
                    temperature=request.temperature,
                    domain_id=request.domain_id,
                )

                return ChatResponse(
                    answer=result["full_text"],
                    steps=result["steps"],
                    jepa_score=result["final_jepa_loss"],
                    mode="nudge",
                    metadata={
                        "iterations": result["iterations"],
                        "num_steps": len(result["steps"]),
                    },
                )

            elif request.mode == "plan":
                # Plan mode (auto-complete)
                result = auto_complete_missing_steps(
                    prompt=request.prompt,
                    llm=llm,
                    rjepa_client=rjepa_client,
                    num_expected_steps=5,
                    domain_id=request.domain_id,
                )

                return ChatResponse(
                    answer=result["full_text"],
                    steps=result["completed"],
                    mode="plan",
                    metadata={
                        "outline_steps": len(result["outline"]),
                        "completed_steps": len(result["completed"]),
                    },
                )

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown mode: {request.mode}",
                )

        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=str(e),
            )

    @app.post("/api/feedback")
    async def submit_feedback(request: FeedbackRequest):
        """
        Submit user feedback.

        Logs feedback for later processing (apprentissage continu).
        """
        # Log to file for later ingestion
        import json
        from pathlib import Path

        feedback_dir = Path("logs/interactions")
        feedback_dir.mkdir(parents=True, exist_ok=True)

        feedback_data = {
            "session_id": request.session_id,
            "prompt": request.prompt,
            "answer": request.answer,
            "feedback": request.feedback,
            "jepa_score": request.jepa_score,
            "timestamp": datetime.now().isoformat(),
        }

        feedback_file = feedback_dir / f"feedback_{request.session_id}.json"
        with open(feedback_file, "a") as f:
            f.write(json.dumps(feedback_data) + "\n")

        logger.info(f"Feedback logged: {request.feedback} for session {request.session_id}")

        return {"status": "ok", "message": "Feedback received"}

    @app.get("/api/jobs", response_model=List[JobStatus])
    async def get_jobs():
        """
        Get status of all jobs (Prefect integration).

        Returns list of jobs with their status.
        """
        # TODO: Integrate with Prefect API
        # For now, return mock data

        jobs = [
            JobStatus(
                job_id="job-001",
                job_type="teacher_generation",
                status="running",
                progress=0.45,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={"num_problems": 1000, "completed": 450},
            ),
            JobStatus(
                job_id="job-002",
                job_type="build_latents",
                status="queued",
                progress=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={"llm": "qwen3-8b", "split": "train"},
            ),
        ]

        return jobs

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """
        WebSocket endpoint for streaming chat.

        Permet de streamer les tokens au fur et à mesure.
        """
        await websocket.accept()

        try:
            while True:
                # Receive message
                data = await websocket.receive_json()

                prompt = data.get("prompt")
                mode = data.get("mode", "off")

                logger.info(f"WebSocket chat: mode={mode}, prompt={prompt[:100]}...")

                # Send initial response
                await websocket.send_json({
                    "type": "start",
                    "mode": mode,
                })

                # Generate (simplifié pour MVP)
                # TODO: Streamer les tokens au fur et à mesure
                if llm is not None:
                    result = llm.generate_with_cot(
                        prompt=prompt,
                        max_new_tokens=512,
                        temperature=0.7,
                        num_samples=1,
                    )[0]

                    # Send steps progressivement
                    for i, step in enumerate(result["steps"]):
                        await websocket.send_json({
                            "type": "step",
                            "step_idx": i,
                            "text": step,
                        })

                    # Send final
                    await websocket.send_json({
                        "type": "complete",
                        "answer": result["full_text"],
                    })

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")

    return app


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run UI backend gateway")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8300,
        help="Port to bind to",
    )
    parser.add_argument(
        "--student-llm-url",
        type=str,
        default="http://localhost:8000",
        help="Student LLM service URL",
    )
    parser.add_argument(
        "--rjepa-url",
        type=str,
        default="http://localhost:8100",
        help="R-JEPA service URL",
    )

    args = parser.parse_args()

    app = create_app(
        student_llm_url=args.student_llm_url,
        rjepa_service_url=args.rjepa_url,
    )

    logger.info(f"Starting UI backend on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
