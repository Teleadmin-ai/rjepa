# 🐳 Docker Compose — Architecture Complète

Documentation détaillée de l'orchestration Docker pour R-JEPA.

## Sommaire

1. [Vue d'ensemble des services](#vue-densemble-des-services)
2. [docker-compose.yml](#docker-composeyml)
3. [docker-compose.dev.yml](#docker-composedevyml)
4. [Usage](#usage)
5. [Accès aux services](#accès-aux-services)

---

## Vue d'ensemble des services

**OBJECTIF** : Tous les services dans des conteneurs Docker, orchestrés par docker-compose.
Windows + NVIDIA GPU → utilise nvidia-docker runtime.

┌─────────────────────────────────────────────────────────────────────────────┐
│ SERVICES                                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. student-llm       : Serveur vLLM avec Qwen3-8B + extraction latents     │
│ 2. rjepa-service     : API R-JEPA (score, predict, correct)                │
│ 3. teacher-orch      : Teacher orchestrator (généra + valida)              │
│ 4. data-pipeline     : Prefect server + workers pour jobs                  │
│ 5. ui-backend        : Gateway FastAPI (WebSocket, auth)                   │
│ 6. ui-frontend       : Next.js app (dev ou build prod)                     │
│ 7. duckdb            : Service DuckDB (queries sur parquet)                │
│ 8. prefect-server    : Prefect UI (monitoring jobs)                        │
│ 9. wandb-local       : (Optionnel) Instance W&B locale si offline          │
└─────────────────────────────────────────────────────────────────────────────┘

**RÉSEAU** : Tous les services sur réseau Docker "rjepa-network" (bridge).
**VOLUMES** : Partagés entre services pour data/, logs/, checkpoints/.

---

## docker-compose.yml

```yaml
version: '3.8'

services:
  # ═══════════════════════════════════════════════════════════════════════════
  # 1. STUDENT LLM (vLLM server avec Qwen3-8B)
  # ═══════════════════════════════════════════════════════════════════════════
  student-llm:
    build:
      context: .
      dockerfile: docker/student-llm.Dockerfile
    image: rjepa/student-llm:latest
    container_name: rjepa-student-llm
    restart: unless-stopped

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_NAME=Qwen/Qwen3-8B-Instruct
      - QUANTIZATION=awq-4bit
      - MAX_MODEL_LEN=4096
      - GPU_MEMORY_UTILIZATION=0.85
      - LAYER_TO_EXTRACT=-2          # Avant-dernière couche

    ports:
      - "8000:8000"                   # vLLM OpenAI-compatible API
      - "8001:8001"                   # Latent extraction API (custom)

    volumes:
      - ./data:/app/data
      - ./logs/student-llm:/app/logs
      - huggingface_cache:/root/.cache/huggingface

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

    networks:
      - rjepa-network

  # ═══════════════════════════════════════════════════════════════════════════
  # 2. R-JEPA SERVICE (inference API)
  # ═══════════════════════════════════════════════════════════════════════════
  rjepa-service:
    build:
      context: .
      dockerfile: docker/rjepa.Dockerfile
    image: rjepa/rjepa-service:latest
    container_name: rjepa-service
    restart: unless-stopped

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    environment:
      - CUDA_VISIBLE_DEVICES=0
      - RJEPA_CHECKPOINT=/app/data/checkpoints/rjepa-qwen3-8b/latest.pth
      - RJEPA_CONFIG=/app/configs/rjepa/base.yaml

    ports:
      - "8100:8100"                   # R-JEPA API

    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
      - ./logs/rjepa:/app/logs

    depends_on:
      - student-llm

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8100/health"]
      interval: 30s
      timeout: 10s
      retries: 3

    networks:
      - rjepa-network

  # ═══════════════════════════════════════════════════════════════════════════
  # 3. TEACHER ORCHESTRATOR
  # ═══════════════════════════════════════════════════════════════════════════
  teacher-orch:
    build:
      context: .
      dockerfile: docker/teacher-orch.Dockerfile
    image: rjepa/teacher-orch:latest
    container_name: rjepa-teacher-orch
    restart: unless-stopped

    environment:
      - TEACHER_CLAUDE_BASE_URL=${TEACHER_CLAUDE_BASE_URL}
      - TEACHER_CLAUDE_API_KEY=${TEACHER_CLAUDE_API_KEY}
      - TEACHER_CLAUDE_MODEL=${TEACHER_CLAUDE_MODEL}
      - TEACHER_GPT_BASE_URL=${TEACHER_GPT_BASE_URL}
      - TEACHER_GPT_API_KEY=${TEACHER_GPT_API_KEY}
      - TEACHER_GPT_MODEL=${TEACHER_GPT_MODEL}
      - TEACHER_MAX_BUDGET_PER_JOB=${TEACHER_MAX_BUDGET_PER_JOB:-50.0}

    ports:
      - "8200:8200"                   # Teacher API

    volumes:
      - ./data:/app/data
      - ./configs/teacher:/app/configs
      - ./logs/teacher:/app/logs

    networks:
      - rjepa-network

  # ═══════════════════════════════════════════════════════════════════════════
  # 4. DATA PIPELINE (Prefect worker)
  # ═══════════════════════════════════════════════════════════════════════════
  data-pipeline:
    build:
      context: .
      dockerfile: docker/data-pipeline.Dockerfile
    image: rjepa/data-pipeline:latest
    container_name: rjepa-data-pipeline
    restart: unless-stopped

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    environment:
      - PREFECT_API_URL=http://prefect-server:4200/api
      - CUDA_VISIBLE_DEVICES=0

    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
      - ./logs/pipeline:/app/logs

    depends_on:
      - prefect-server
      - student-llm
      - teacher-orch

    command: prefect agent start -q default

    networks:
      - rjepa-network

  # ═══════════════════════════════════════════════════════════════════════════
  # 5. PREFECT SERVER (orchestration UI)
  # ═══════════════════════════════════════════════════════════════════════════
  prefect-server:
    image: prefecthq/prefect:2-python3.11
    container_name: rjepa-prefect-server
    restart: unless-stopped

    ports:
      - "4200:4200"                   # Prefect UI

    environment:
      - PREFECT_SERVER_API_HOST=0.0.0.0
      - PREFECT_API_DATABASE_CONNECTION_URL=sqlite:///prefect.db

    volumes:
      - prefect_data:/root/.prefect

    command: prefect server start --host 0.0.0.0

    networks:
      - rjepa-network

  # ═══════════════════════════════════════════════════════════════════════════
  # 6. UI BACKEND (Gateway FastAPI + WebSocket)
  # ═══════════════════════════════════════════════════════════════════════════
  ui-backend:
    build:
      context: .
      dockerfile: docker/ui-backend.Dockerfile
    image: rjepa/ui-backend:latest
    container_name: rjepa-ui-backend
    restart: unless-stopped

    environment:
      - STUDENT_LLM_URL=http://student-llm:8000
      - RJEPA_SERVICE_URL=http://rjepa-service:8100
      - PREFECT_API_URL=http://prefect-server:4200/api

    ports:
      - "8300:8300"                   # UI backend API

    volumes:
      - ./logs/interactions:/app/logs/interactions

    depends_on:
      - student-llm
      - rjepa-service
      - prefect-server

    networks:
      - rjepa-network

  # ═══════════════════════════════════════════════════════════════════════════
  # 7. UI FRONTEND (Next.js)
  # ═══════════════════════════════════════════════════════════════════════════
  ui-frontend:
    build:
      context: ./ui/web
      dockerfile: ../../docker/ui-frontend.Dockerfile
    image: rjepa/ui-frontend:latest
    container_name: rjepa-ui-frontend
    restart: unless-stopped

    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8300

    ports:
      - "3000:3000"                   # Next.js app

    depends_on:
      - ui-backend

    networks:
      - rjepa-network

# ═══════════════════════════════════════════════════════════════════════════
# VOLUMES PARTAGÉS
# ═══════════════════════════════════════════════════════════════════════════
volumes:
  huggingface_cache:                # Cache modèles HF (persistant)
  prefect_data:                     # DB Prefect

# ═══════════════════════════════════════════════════════════════════════════
# RÉSEAU
# ═══════════════════════════════════════════════════════════════════════════
networks:
  rjepa-network:
    driver: bridge
```

---

## docker-compose.dev.yml

Override pour développement local avec hot reload.

**Utiliser:** `docker-compose -f docker-compose.yml -f docker-compose.dev.yml up`

```yaml
version: '3.8'

services:
  student-llm:
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./rjepa:/app/rjepa:ro      # Mount code en lecture seule pour hot reload

  rjepa-service:
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./rjepa:/app/rjepa:ro

  ui-frontend:
    command: npm run dev             # Mode dev Next.js (hot reload)
    volumes:
      - ./ui/web:/app:delegated      # Mount UI code pour hot reload
```

---

## Usage

```bash
# Build toutes les images
make docker-build
# ou: docker-compose build

# Lancer tous les services (prod)
make docker-up
# ou: docker-compose up -d

# Lancer en mode dev (avec hot reload)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Voir les logs
make docker-logs
# ou: docker-compose logs -f

# Arrêter
make docker-down
# ou: docker-compose down

# Rebuild un seul service
docker-compose build student-llm
docker-compose up -d student-llm
```

---

## Accès aux services

| Service | URL | Description |
|---------|-----|-------------|
| Chat UI | http://localhost:3000 | Interface utilisateur Next.js |
| Prefect UI | http://localhost:4200 | Monitoring des jobs |
| Student LLM API | http://localhost:8000 | API vLLM OpenAI-compatible |
| R-JEPA API | http://localhost:8100 | API inference R-JEPA |
| Teacher API | http://localhost:8200 | API Teacher orchestrator |
| UI Backend | http://localhost:8300 | Gateway FastAPI |
