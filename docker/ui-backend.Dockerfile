# UI Backend Gateway Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (subset for UI backend)
COPY pyproject.toml ./
RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx python-multipart websockets pydantic-settings

# Copy R-JEPA package (for imports)
COPY rjepa/ ./rjepa/

# Copy UI server code
COPY ui/server/ ./ui/server/

# Expose port
EXPOSE 8300

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8300/health || exit 1

# Run UI backend server
CMD ["python", "-m", "uvicorn", "ui.server.main:app", "--host", "0.0.0.0", "--port", "8300"]
