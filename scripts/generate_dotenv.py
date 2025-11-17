#!/usr/bin/env python3
"""
Interactive .env file generator for R-JEPA.
"""
import os
from pathlib import Path


def prompt(question, default=None):
    """Prompt user for input with optional default."""
    if default:
        response = input(f"{question} [{default}]: ").strip()
        return response if response else default
    else:
        return input(f"{question}: ").strip()


def main():
    print("=" * 80)
    print("R-JEPA .env Generator")
    print("=" * 80 + "\n")

    env_file = Path(".env")
    if env_file.exists():
        response = prompt("⚠️  .env already exists. Overwrite? [y/N]", "n").lower()
        if response not in ['y', 'yes']:
            print("Cancelled.")
            return

    print("Let's configure your R-JEPA environment!\n")

    # Teacher APIs
    print("─" * 80)
    print("TEACHER LLM APIs (OpenAI-compatible)")
    print("─" * 80)

    teacher_claude_url = prompt("Claude API base URL", "http://localhost:8001/v1")
    teacher_claude_key = prompt("Claude API key", "sk-xxx-replace-me")
    teacher_claude_model = prompt("Claude model", "claude-3-5-sonnet-20241022")

    teacher_gpt_url = prompt("GPT API base URL", "http://localhost:8002/v1")
    teacher_gpt_key = prompt("GPT API key", "sk-xxx-replace-me")
    teacher_gpt_model = prompt("GPT model", "gpt-4-turbo-2024-04-09")

    teacher_budget = prompt("Max budget per job (USD)", "50.0")

    # Student LLM
    print("\n" + "─" * 80)
    print("STUDENT LLM Configuration")
    print("─" * 80)

    student_model = prompt("Student model name", "Qwen/Qwen3-8B-Instruct")
    student_quant = prompt("Quantization", "awq-4bit")
    student_layer = prompt("Layer to extract", "-2")

    # Tracking
    print("\n" + "─" * 80)
    print("TRACKING & MONITORING")
    print("─" * 80)

    wandb_key = prompt("W&B API key (leave empty to skip)", "")
    wandb_project = prompt("W&B project", "rjepa-training")
    wandb_entity = prompt("W&B entity", "teleadmin-ai")

    # Build .env content
    env_content = f"""# ═══════════════════════════════════════════════════════════════════════════════
# R-JEPA Configuration (auto-generated)
# ═══════════════════════════════════════════════════════════════════════════════

# TEACHER LLM APIs
TEACHER_CLAUDE_BASE_URL={teacher_claude_url}
TEACHER_CLAUDE_API_KEY={teacher_claude_key}
TEACHER_CLAUDE_MODEL={teacher_claude_model}

TEACHER_GPT_BASE_URL={teacher_gpt_url}
TEACHER_GPT_API_KEY={teacher_gpt_key}
TEACHER_GPT_MODEL={teacher_gpt_model}

TEACHER_MAX_BUDGET_PER_JOB={teacher_budget}

# STUDENT LLM
STUDENT_MODEL_NAME={student_model}
STUDENT_QUANTIZATION={student_quant}
STUDENT_LAYER_TO_EXTRACT={student_layer}
STUDENT_MAX_MODEL_LEN=4096
STUDENT_GPU_MEMORY_UTILIZATION=0.85

# TRACKING
WANDB_API_KEY={wandb_key}
WANDB_PROJECT={wandb_project}
WANDB_ENTITY={wandb_entity}

# R-JEPA MODEL
RJEPA_DIM=1024
RJEPA_DEPTH_ENC=12
RJEPA_DEPTH_PRED=8
RJEPA_NUM_HEADS=16
RJEPA_DOMAIN_EMBED_DIM=64

# TRAINING
TRAIN_BATCH_SIZE=64
TRAIN_LR=3e-4
TRAIN_EMA_MOMENTUM_START=0.996
TRAIN_EMA_MOMENTUM_END=1.0
TRAIN_EPOCHS=10
TRAIN_AMP=bf16
TRAIN_GRAD_CLIP=1.0

# DATA PATHS
DATA_ROOT=./data
DATASETS_ROOT=./data/datasets
LATENTS_ROOT=./data/latents
CHECKPOINTS_ROOT=./data/checkpoints
LOGS_ROOT=./logs

# PREFECT
PREFECT_API_URL=http://localhost:4200/api
PREFECT_API_DATABASE_CONNECTION_URL=sqlite:///prefect.db

# SERVICES URLs
STUDENT_LLM_URL=http://localhost:8000
RJEPA_SERVICE_URL=http://localhost:8100
TEACHER_ORCH_URL=http://localhost:8200
UI_BACKEND_URL=http://localhost:8300

# USER FEEDBACK
ENABLE_USER_FEEDBACK_LOOP=true
ANONYMIZE_USER_DATA=true
PII_FILTER_ENABLED=true

# DEVELOPMENT
LOG_LEVEL=INFO
DEBUG=false
ENVIRONMENT=development
"""

    # Write .env
    with open(".env", "w") as f:
        f.write(env_content)

    print("\n" + "=" * 80)
    print("✅ .env file created successfully!")
    print("=" * 80)
    print("\nYou can now:")
    print("  - Edit .env to customize settings")
    print("  - Run: make setup")
    print("  - Run: make docker-up")


if __name__ == "__main__":
    main()
