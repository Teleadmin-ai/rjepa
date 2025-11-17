# Data Pipeline Dockerfile (Prefect worker + GPU support)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy requirements
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir transformers accelerate safetensors
RUN pip install --no-cache-dir prefect pyarrow duckdb datasets

# Copy R-JEPA package
COPY rjepa/ ./rjepa/
COPY configs/ ./configs/

# Install R-JEPA package
RUN pip install --no-cache-dir -e .

# Prefect agent command (will be overridden by docker-compose)
CMD ["prefect", "agent", "start", "-q", "default"]
