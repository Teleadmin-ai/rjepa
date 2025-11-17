# ═══════════════════════════════════════════════════════════════════════════════
# Student LLM Dockerfile (vLLM + Qwen3-8B + Latent Extraction)
# ═══════════════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install \
    transformers>=4.38.0 \
    accelerate \
    safetensors \
    fastapi \
    uvicorn[standard] \
    pydantic-settings \
    python-multipart

# Install quantization libraries (AWQ preferred for Qwen3)
RUN pip install autoawq

# Create app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml /app/
COPY rjepa/ /app/rjepa/

# Install R-JEPA package
RUN pip install -e .

# Expose ports
# 8000: vLLM OpenAI-compatible API
# 8001: Latent extraction API
EXPOSE 8000 8001

# Environment variables (can be overridden)
ENV MODEL_NAME="Qwen/Qwen3-8B-Instruct"
ENV QUANTIZATION="awq-4bit"
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.85
ENV LAYER_TO_EXTRACT=-2
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
CMD ["python", "-m", "rjepa.llm.server"]
