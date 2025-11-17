"""
R-JEPA Configuration Settings (Pydantic).
"""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Global configuration settings for R-JEPA.
    Loaded from .env file or environment variables.
    """

    # ───────────────────────────────────────────────────────────────────────────────
    # TEACHER LLM APIs (OpenAI-compatible loopback)
    # ───────────────────────────────────────────────────────────────────────────────

    # Teacher 1: Claude (via proxy)
    teacher_claude_base_url: str = "http://localhost:8001/v1"
    teacher_claude_api_key: Optional[str] = None
    teacher_claude_model: str = "claude-3-5-sonnet-20241022"

    # Teacher 2: GPT (via proxy)
    teacher_gpt_base_url: str = "http://localhost:8002/v1"
    teacher_gpt_api_key: Optional[str] = None
    teacher_gpt_model: str = "gpt-4-turbo-2024-04-09"

    # Teacher budget limits (USD per job)
    teacher_max_budget_per_job: float = 50.0

    # ───────────────────────────────────────────────────────────────────────────────
    # STUDENT LLM (Qwen3-8B)
    # ───────────────────────────────────────────────────────────────────────────────

    student_model_name: str = "Qwen/Qwen3-8B-Instruct"
    student_quantization: str = "awq-4bit"  # "awq-4bit", "gptq-4bit", or None
    student_layer_to_extract: int = -2  # Layer to extract latents (-2 = second-to-last)
    student_max_model_len: int = 4096
    student_gpu_memory_utilization: float = 0.85

    # ───────────────────────────────────────────────────────────────────────────────
    # TRACKING (W&B)
    # ───────────────────────────────────────────────────────────────────────────────

    wandb_api_key: Optional[str] = None
    wandb_project: str = "rjepa-training"
    wandb_entity: Optional[str] = None

    # ───────────────────────────────────────────────────────────────────────────────
    # PREFECT (Orchestration)
    # ───────────────────────────────────────────────────────────────────────────────

    prefect_api_url: str = "http://prefect-server:4200/api"

    # ───────────────────────────────────────────────────────────────────────────────
    # DATA PATHS
    # ───────────────────────────────────────────────────────────────────────────────

    data_root: str = "./data"
    logs_root: str = "./logs"
    checkpoints_root: str = "./data/checkpoints"

    # ───────────────────────────────────────────────────────────────────────────────
    # MISC
    # ───────────────────────────────────────────────────────────────────────────────

    seed: int = 42
    device: str = "cuda"

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
