# AuthentiGuard — AI Authenticity Detection Platform

> Multi-modal AI content detection: text, image, video, audio, and code.

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/your-org/authentiguard.git
cd authentiguard

# 2. Copy environment template and fill in all values
cp .env.example .env

# 3. Start the full local stack
docker compose up --build

# Services:
#   Frontend  → http://localhost:3000
#   Backend   → http://localhost:8000
#   API docs  → http://localhost:8000/docs
#   MLflow    → http://localhost:5000
#   Flower    → http://localhost:5555
#   MinIO     → http://localhost:9001
```

## Monorepo Structure

```
authentiguard/
├── frontend/          Next.js + React + TypeScript + Tailwind
├── backend/           FastAPI — API gateway, auth, upload, orchestration
├── ai/
│   ├── text-detector/
│   ├── audio-detector/
│   ├── video-detector/
│   ├── image-detector/
│   ├── code-detector/
│   └── ensemble-engine/
├── datasets/
│   ├── human/         Human-authored content
│   ├── ai-generated/  AI-generated content
│   └── adversarial/   Attack/evasion samples
├── infra/
│   ├── docker/        Compose configs, init scripts
│   ├── k8s/           Helm charts, manifests
│   └── terraform/     IaC for cloud infra
├── security/          Encryption, signatures, compliance
└── docs/              API docs, architecture, onboarding
```

## Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Production — only merged from `develop` via PR |
| `develop` | Integration — feature branches merge here |
| `feature/*` | Individual features / detectors |

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11, SQLAlchemy 2, Alembic |
| Queue | Celery + Redis 7 |
| Database | PostgreSQL 16 |
| Storage | S3 / Cloudflare R2 (MinIO locally) |
| ML | PyTorch, Transformers, scikit-learn, ONNX |
| Tracking | MLflow / Weights & Biases |
| Data versioning | DVC |
| Infra | Docker, Kubernetes, Helm, Terraform |

## Documentation

- [API Reference](docs/api/)
- [Architecture](docs/architecture/)
- [Developer Onboarding](docs/onboarding/)
- [Dataset Documentation](datasets/)
