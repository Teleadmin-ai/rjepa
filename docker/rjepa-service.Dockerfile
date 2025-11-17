# ═══════════════════════════════════════════════════════════════════════════════
# R-JEPA Inference Service Dockerfile
# ═══════════════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Éviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 + dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml /app/
COPY rjepa/ /app/rjepa/
COPY configs/ /app/configs/

# Install dependencies (server extras only)
RUN pip install -e ".[server]" --no-cache-dir

# Install additional dependencies for inference service
RUN pip install \
    fastapi \
    uvicorn[standard] \
    httpx \
    pydantic \
    --no-cache-dir

# Expose port for R-JEPA service
EXPOSE 8100

# Default environment variables
ENV RJEPA_CHECKPOINT=/app/data/checkpoints/rjepa-qwen3-8b/latest.pth
ENV RJEPA_DEVICE=cuda
ENV RJEPA_HOST=0.0.0.0
ENV RJEPA_PORT=8100

# Volume for data (checkpoints)
VOLUME ["/app/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

# Run R-JEPA service
CMD ["python", "-m", "rjepa.jepa.service", \
     "--checkpoint", "${RJEPA_CHECKPOINT}", \
     "--device", "${RJEPA_DEVICE}", \
     "--host", "${RJEPA_HOST}", \
     "--port", "${RJEPA_PORT}"]
