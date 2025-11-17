# ═══════════════════════════════════════════════════════════════════════════════
# Teacher Orchestrator Dockerfile
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml /app/
COPY rjepa/ /app/rjepa/
COPY configs/ /app/configs/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -e ".[train]"

# Install additional dependencies for teacher
RUN pip install \
    sympy \
    prefect>=2.0

# Environment variables (can be overridden)
ENV TEACHER_CLAUDE_BASE_URL="http://localhost:8001/v1"
ENV TEACHER_GPT_BASE_URL="http://localhost:8002/v1"
ENV TEACHER_MAX_BUDGET_PER_JOB=50.0

# Expose port (for monitoring/API if needed)
EXPOSE 8200

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import rjepa.teacher; print('OK')" || exit 1

# Default command: wait for Prefect agent start
CMD ["prefect", "agent", "start", "-q", "default"]
