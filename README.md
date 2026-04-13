# AuthentiGuard — AI Authenticity Detection Platform

AuthentiGuard is an adversarial AI detection platform that verifies whether content is human or AI-generated.

As AI models become indistinguishable from humans, existing detectors fail — they rely on surface patterns that break with simple edits.

We take a different approach:
we train on adversarial datasets where AI and human content are intentionally made similar, forcing the model to learn real underlying signals instead of shortcuts.

Our system achieves 95%+ accuracy on balanced adversarial data and significantly outperforms traditional detectors on real-world cases where AI tries to pass as human.

We’re building the API layer for content verification, enabling platforms, schools, and businesses to restore trust in digital content.


## Monorepo Structure

```
authentiguard/
├── frontend/          Next.js + React + TypeScript + Tailwind
├── backend/           FastAPI — API gateway, auth, upload, orchestration
├── ai/
│   ├── text_detector/
│   ├── audio_detector/
│   ├── video_detector/
│   ├── image_detector/
│   ├── code_detector/
│   └── ensemble_engine/
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
