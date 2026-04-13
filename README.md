# AuthentiGuard — AI Content Detection Platform

AAuthentiGuard is an adversarial AI detection platform that verifies whether content is human or AI-generated.

As AI models become indistinguishable from humans, existing detectors fail — they rely on surface patterns that break with simple edits.

We take a different approach:
we train on adversarial datasets where AI and human content are intentionally made similar, forcing the model to learn real underlying signals instead of shortcuts.

Our system achieves 95%+ accuracy on balanced adversarial data and significantly outperforms traditional detectors on real-world cases where AI tries to pass as human.

We’re building the API layer for content verification, enabling platforms, schools, and businesses to restore trust in digital content.

## Monorepo Structure

```
authentiguard/
├── frontend/          Next.js 14 + React + TypeScript + Tailwind
├── backend/           FastAPI — API gateway, auth, upload, orchestration
├── ai/
│   ├── text_detector/     Production text detection ensemble
│   ├── image_detector/    Experimental
│   ├── audio_detector/    Experimental
│   ├── video_detector/    Experimental
│   ├── code_detector/     Experimental
│   └── ensemble_engine/   Detection routing + dispatch
├── datasets/
├── scripts/           Training + evaluation scripts
├── infra/
│   ├── docker/        Compose configs
│   ├── k8s/           Kubernetes manifests
│   └── terraform/     AWS IaC
└── security/          Encryption, signatures, compliance
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11, SQLAlchemy 2, Alembic |
| Queue | Celery + Redis 7 |
| Database | PostgreSQL 16 |
| Storage | S3 / MinIO |
| ML | PyTorch, Transformers, scikit-learn |
| Infra | Docker, Kubernetes, Terraform |


## API

- `POST /api/v1/analyze/text` — Submit text for detection
- `POST /api/v1/analyze/file` — Upload file for detection
- `POST /api/v1/analyze/url` — Submit URL for content fetch + detection
- `GET  /api/v1/jobs/{id}` — Poll job status
- `GET  /api/v1/jobs/{id}/result` — Get detection result
- `GET  /api/v1/jobs/{id}/report` — Export report (JSON/PDF)
