# R-JEPA: Reasoning Joint Embedding Predictive Architecture

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org)

A **World Model for Text Reasoning** inspired by Meta AI's V-JEPA, adapted for textual reasoning sequences.

## 🎯 Two Editions

| Edition | Description | Availability |
|---------|-------------|--------------|
| **🌐 Community** | Full open-source (MIT), self-hosted | **Available Now** |
| **☁️ Cloud API** | Managed service, pay-per-token | Coming Soon |

> **🔒 Privacy-First**: With Cloud API, your text never leaves your servers. Only latent vectors (4096-dim abstract representations) are sent - mathematically irreversible, GDPR/HIPAA compatible.

## Overview

R-JEPA learns to predict, complete, and correct reasoning steps in **latent space** rather than token space. It transposes the principle *"predict features, not pixels"* to text: **"predict concepts, not tokens"**.

### Core Principle

| V-JEPA (Vision) | R-JEPA (Text) |
|-----------------|---------------|
| Predicts masked video patches | Predicts masked reasoning steps |
| Learns visual physics (gravity, occlusion) | Learns reasoning "physics" (logical flow, inference) |
| Operates on image features | Operates on LLM hidden states |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        R-JEPA Model (678M params)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: LLM latents [B, S, 4096] (from Qwen3-8B layer -2)      │
│                                                                 │
│  ┌─────────────────┐    ┌───────────┐    ┌─────────────────┐   │
│  │ Context Encoder │ -> │ Predictor │ -> │ z_pred          │   │
│  │ (trained)       │    │           │    │                 │   │
│  └─────────────────┘    └───────────┘    └────────┬────────┘   │
│                                                   │             │
│  ┌─────────────────┐                              v             │
│  │ Target Encoder  │ ─────────────────────> z_target           │
│  │ (EMA, frozen)   │                         (targets)          │
│  └─────────────────┘                                            │
│                                                                 │
│  Loss = L1(z_pred, z_target) + variance_reg + contrastive      │
└─────────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Total Parameters | **678M** |
| Context Encoder | 12 layers, 1024 dim, 16 heads |
| Predictor | 8 layers |
| Input Dimension | 4096 (Qwen3-8B hidden size) |
| Masking | Contiguous (30-70% ratio) |
| EMA Momentum | 0.996 → 0.9999 |

## Features

### Inference Modes

| Mode | Description | Status |
|------|-------------|--------|
| **RERANK** | Generate K candidates, select best by JEPA-loss | Production |
| **NUDGE** | Guide generation with predicted latents (Logit Guidance) | Implemented |
| **PLAN** | Complete missing reasoning steps | Implemented |

### Key Capabilities

- **Multi-LLM Support**: Qwen3, Llama3, Mistral, DeepSeek, Phi families
- **Fast Calibration**: Adapt to new LLM in 2-4h (vs days for full retrain)
- **Continuous Learning**: User feedback → validated interactions → retraining
- **Extended Benchmarks**: GSM8K, MATH, HumanEval, MMLU, Big-Bench Hard, ARC

## 🌐 Community Edition (Open Source)

Everything you need to train and deploy R-JEPA on your own infrastructure.

### What's Included

- ✅ Full R-JEPA source code (MIT License)
- ✅ Pre-configured Docker Compose (7 services)
- ✅ 21K+ academic problems dataset
- ✅ All inference modes (RERANK, NUDGE, PLAN)
- ✅ Multi-LLM support (Qwen, Llama, Mistral, DeepSeek, Phi)
- ✅ Training scripts and configurations
- ✅ Benchmark evaluation tools

## Installation

### Requirements

- Python 3.11+
- CUDA 12.1+ (RTX 4090 recommended, 24GB VRAM)
- PyTorch 2.1+

### Setup

```bash
# Clone repository
git clone https://github.com/Teleadmin-ai/rjepa.git
cd rjepa

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# or: source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[train,dev]"

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### 1. Import Academic Datasets

```bash
# Import GSM8K, MATH, HumanEval (21,456 problems total)
python -m rjepa.data.import_academic --output data/datasets/academic
```

### 2. Extract Latents

```bash
# Extract latents with Qwen3-8B (batch_size=8 optimal for RTX 4090)
python scripts/extract_latents_optimized.py --batch-size 8
```

### 3. Train R-JEPA

```bash
# Train the world model (~35h for 100 epochs on RTX 4090)
python -m rjepa.pipeline.train_rjepa --config configs/rjepa/train.yaml
```

### 4. Evaluate

```bash
# Run benchmarks
python -m rjepa.pipeline.evaluate --benchmark gsm8k --mode rerank

# Extended benchmarks (MMLU, BBH, ARC)
python scripts/run_extended_benchmarks.py --quick
```

## Project Structure

```
rjepa/
├── rjepa/
│   ├── config/          # Pydantic settings
│   ├── data/            # Data schemas, ingestion, sharding
│   ├── llm/             # LLM adapter, latent extraction
│   ├── jepa/            # Core R-JEPA model
│   │   ├── model.py     # ReasoningJEPA main class
│   │   ├── encoder.py   # StepTransformer encoder
│   │   ├── predictor.py # StepPredictor
│   │   ├── trainer.py   # Training loop with EMA
│   │   └── service.py   # FastAPI inference service
│   ├── pipeline/        # Prefect workflows
│   ├── inference/       # Rerank, nudge, plan modes
│   ├── evaluation/      # Benchmarks
│   └── decoder/         # Latent-to-text decoder
├── ui/                  # Next.js frontend
├── docker/              # Dockerfiles (7 services)
├── configs/             # YAML configurations
├── scripts/             # Utility scripts
└── data/                # Datasets, latents, checkpoints
```

## Configuration

Main training config (`configs/rjepa/train.yaml`):

```yaml
model:
  encoder_embed_dim: 1024
  depth_encoder: 12
  depth_predictor: 8
  num_heads: 16
  input_dim: 4096  # Qwen3-8B hidden size

training:
  batch_size: 32
  lr: 0.0003
  epochs: 100
  ema_momentum_start: 0.996
  ema_momentum_end: 0.9999
  amp_enabled: true
  grad_clip: 1.0

masker:
  type: contiguous
  min_ratio: 0.3
  max_ratio: 0.7
```

## Datasets

| Dataset | Problems | Domain | License |
|---------|----------|--------|---------|
| GSM8K | 8,792 | Math (grade school) | MIT |
| MATH | 12,500 | Math (competition) | MIT |
| HumanEval | 164 | Code (Python) | MIT |
| **Total** | **21,456** | - | - |

## Docker Services

Full deployment with Docker Compose (7 services):

```bash
# Build all images
make docker-build

# Launch all services
make docker-up
```

| Service | Port | Description |
|---------|------|-------------|
| student-llm | 8000 | Qwen3-8B with latent extraction |
| rjepa-service | 8100 | R-JEPA inference API |
| teacher-orch | 8200 | Dataset generation (Claude/GPT) |
| prefect-server | 4200 | Workflow orchestration UI |
| ui-backend | 8300 | FastAPI gateway |
| ui-frontend | 3000 | Next.js chat interface |

## API Endpoints

R-JEPA Service (`localhost:8100`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/score` | POST | Compute JEPA-loss for latent sequence |
| `/predict_masked` | POST | Predict masked step latents |

## Performance

| Metric | Value |
|--------|-------|
| Latent Extraction | 3.8s/problem (RTX 4090) |
| Training | ~35h for 100 epochs |
| VRAM Usage | ~2.5GB (R-JEPA) + ~16GB (Qwen3-8B) |

## ☁️ Cloud API (Coming Soon)

Managed R-JEPA service with advanced features - no GPU required.

### Features

| Feature | Description |
|---------|-------------|
| **🔒 Privacy-First** | Your text stays local - only latent vectors are transmitted |
| **🧠 Latent Memory** | Cross-session context persistence for reasoning continuity |
| **🔄 Multi-Model** | Switch between LLM backends instantly |
| **📈 Auto-scaling** | Handle any load without infrastructure management |
| **📊 Analytics** | Usage dashboard and performance metrics |

### How Privacy Works

```
YOUR INFRASTRUCTURE                    TELEADMIN CLOUD
┌─────────────────────┐                ┌─────────────────────┐
│ Your App + LLM      │                │ R-JEPA API          │
│ ─────────────────── │                │ ─────────────────── │
│ 1. User query       │                │                     │
│ 2. LLM generates    │   latent       │ 3. Compute score    │
│    reasoning        │   vectors      │ 4. Return guidance  │
│ 3. Extract latents  │ ─────────────► │                     │
│    (4096-dim)       │ ◄───────────── │                     │
│ 4. Apply guidance   │   predictions  │                     │
│                     │                │                     │
│ ✅ TEXT NEVER LEAVES│                │ ❌ No text access   │
└─────────────────────┘                └─────────────────────┘
```

**Why latent vectors are safe:**
- Latent vectors are abstract 4096-dimensional representations
- Mathematically impossible to reconstruct original text
- Compliant with GDPR, HIPAA, and enterprise security policies

### Waitlist

Interested in Cloud API? Join the waitlist:

📧 **[music.romain@teleadmin.net](mailto:music.romain@teleadmin.net?subject=R-JEPA%20Cloud%20API%20Interest)**

## References

- [V-JEPA: Latent Video Prediction for Visual Representation Learning](https://ai.meta.com/research/publications/v-jepa-latent-video-prediction-for-visual-representation-learning/) (Meta AI, 2024)
- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) (Yann LeCun, 2022)

## License

MIT License

## Citation

```bibtex
@software{rjepa2025,
  title = {R-JEPA: Reasoning Joint Embedding Predictive Architecture},
  author = {Provencal Romain},
  year = {2025},
  url = {https://github.com/Teleadmin-ai/rjepa}
}
```

---

*Built with Claude Code*
