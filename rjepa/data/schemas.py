"""
Data schemas for R-JEPA (Pydantic models).
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field


class Problem(BaseModel):
    """
    A reasoning problem (math, code, logic, etc.).
    """
    problem_id: str = Field(..., description="Unique problem identifier")
    domain: str = Field(..., description="Domain: 'math', 'code', 'logic'")
    subdomain: str = Field(..., description="Subdomain: 'algebra', 'geometry', 'python', etc.")
    source: str = Field(..., description="Source: 'teacher_claude', 'gsm8k', 'humaneval', etc.")
    difficulty: str = Field(..., description="Difficulty: 'easy', 'medium', 'hard'")
    statement: str = Field(..., description="Problem statement text")
    answer_gold: Optional[str] = Field(None, description="Gold answer if available")
    meta_course: Optional[Dict] = Field(None, description="Course metadata (chapter, notions, etc.)")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "problem_id": "math_001",
                "domain": "math",
                "subdomain": "algebra",
                "source": "teacher_claude",
                "difficulty": "easy",
                "statement": "Solve for x: 2x + 5 = 13",
                "answer_gold": "4",
                "meta_course": {"chapter": "Linear Equations", "notions": ["solving", "simplification"]},
            }
        }


class ChainOfThought(BaseModel):
    """
    A chain of thought (reasoning steps) for a problem.
    """
    cot_id: str = Field(..., description="Unique CoT identifier")
    problem_id: str = Field(..., description="Associated problem ID")
    steps: List[str] = Field(..., description="List of reasoning steps (text)")
    final_answer: str = Field(..., description="Final answer produced")
    is_valid: bool = Field(..., description="Whether the reasoning is valid (passed validation)")
    validation_reason: str = Field(..., description="Reason for validation result")
    teacher_model: str = Field(..., description="Model that generated this CoT (e.g., 'claude-3-5-sonnet')")
    source: str = Field(..., description="Source: 'teacher_claude', 'teacher_gpt', 'user'")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    meta: Optional[Dict] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "cot_id": "cot_001",
                "problem_id": "math_001",
                "steps": [
                    "Step 1: Subtract 5 from both sides: 2x = 8",
                    "Step 2: Divide both sides by 2: x = 4",
                ],
                "final_answer": "4",
                "is_valid": True,
                "validation_reason": "Verified symbolically with sympy",
                "teacher_model": "claude-3-5-sonnet-20241022",
                "source": "teacher_claude",
            }
        }


class LatentSequence(BaseModel):
    """
    Latent representations extracted from a CoT by a specific LLM.

    Note: The actual latent vectors (H) are stored separately in safetensors files.
    This schema only contains metadata.
    """
    latent_id: str = Field(..., description="Unique latent sequence identifier")
    problem_id: str = Field(..., description="Associated problem ID")
    cot_id: str = Field(..., description="Associated CoT ID")
    llm_tag: str = Field(..., description="LLM used to extract latents (e.g., 'qwen3-8b-instruct-awq')")
    layer_idx: int = Field(..., description="Layer index extracted (e.g., -2)")
    hidden_size: int = Field(..., description="Hidden size of the LLM (e.g., 4096 for Qwen3-8B)")
    num_steps: int = Field(..., description="Number of reasoning steps")
    step_boundaries: List[Tuple[int, int]] = Field(
        ..., description="Token boundaries for each step [(start, end), ...]"
    )
    domain: str = Field(..., description="Problem domain")
    subdomain: str = Field(..., description="Problem subdomain")
    safetensors_path: Optional[str] = Field(
        None, description="Path to safetensors file containing H tensor"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    extra: Optional[Dict] = Field(None, description="Extra metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "latent_id": "latent_001",
                "problem_id": "math_001",
                "cot_id": "cot_001",
                "llm_tag": "qwen3-8b-instruct-awq",
                "layer_idx": -2,
                "hidden_size": 4096,
                "num_steps": 2,
                "step_boundaries": [(0, 15), (15, 28)],
                "domain": "math",
                "subdomain": "algebra",
                "safetensors_path": "data/latents/qwen3-8b/train/shard-0000.safetensors",
            }
        }


class DatasetVersion(BaseModel):
    """
    Metadata for a versioned dataset.
    """
    version: str = Field(..., description="Version string (e.g., 'v1.0.0')")
    date: str = Field(..., description="Release date (YYYY-MM-DD)")
    num_problems: int = Field(..., description="Total number of problems")
    num_cots: int = Field(..., description="Total number of CoTs")
    sources: List[str] = Field(..., description="Data sources used")
    validation_rate: float = Field(..., description="Validation pass rate (0.0-1.0)")
    description: Optional[str] = Field(None, description="Version description")

    class Config:
        json_schema_extra = {
            "example": {
                "version": "v1.0.0",
                "date": "2025-01-15",
                "num_problems": 12000,
                "num_cots": 36000,
                "sources": ["teacher_claude", "teacher_gpt", "gsm8k"],
                "validation_rate": 0.89,
                "description": "Initial dataset with math problems",
            }
        }


class RJEPACheckpoint(BaseModel):
    """
    Metadata for an R-JEPA checkpoint.
    """
    checkpoint_id: str = Field(..., description="Unique checkpoint identifier")
    llm_tag: str = Field(..., description="LLM used for training (e.g., 'qwen3-8b')")
    dataset_version: str = Field(..., description="Dataset version used (e.g., 'v1.0.0')")
    epoch: int = Field(..., description="Training epoch")
    train_loss: float = Field(..., description="Training loss")
    val_loss: float = Field(..., description="Validation loss")
    config_path: str = Field(..., description="Path to config YAML")
    checkpoint_path: str = Field(..., description="Path to checkpoint file")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    eval_results: Optional[Dict] = Field(None, description="Evaluation results (benchmarks)")

    class Config:
        json_schema_extra = {
            "example": {
                "checkpoint_id": "rjepa-qwen3-8b-v1.0.0-epoch-10",
                "llm_tag": "qwen3-8b",
                "dataset_version": "v1.0.0",
                "epoch": 10,
                "train_loss": 0.15,
                "val_loss": 0.18,
                "config_path": "configs/rjepa/base.yaml",
                "checkpoint_path": "data/checkpoints/rjepa-qwen3-8b/v1.0.0/checkpoint-epoch-10.pth",
                "eval_results": {"gsm8k": {"accuracy": 0.78}},
            }
        }
