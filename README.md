# AuthentiGuard — AI Content Detection Platform

AI-generated content detection focused on **text detection** as the production-ready core, with experimental support for image, audio, video, and code analysis.

## Current Status

- **Text detection**: Production-ready. DistilBERT ensemble with F1 0.95, AUROC 0.98 (adversarial v3_hard checkpoint).
- **Image/Audio/Video/Code**: Experimental. Models exist but are not production-tested.
- **Passport/Provenance**: Disabled (not production-ready).

See [SUMMARY.md](SUMMARY.md) for detailed status.

AuthentiGuard is an adversarial AI detection platform that verifies whether content is human or AI-generated.

As AI models become indistinguishable from humans, existing detectors fail — they rely on surface patterns that break with simple edits.

We take a different approach:
we train on adversarial datasets where AI and human content are intentionally made similar, forcing the model to learn real underlying signals instead of shortcuts.

Our system achieves ~95% F1 on adversarial data and is designed to outperform traditional detectors on real-world cases where AI tries to pass as human.

We’re building the API layer for content verification, enabling platforms, schools, and businesses to restore trust in digital content.

## Quickstart

### Prerequisites

- Python 3.11+
- Node.js 20+ (with npm 10+)
- Docker and Docker Compose
- GPU: **optional**. The text detector runs on CPU; a CUDA-capable GPU speeds up inference and is recommended for training but not required to run the stack.

### Run the stack

```bash
git clone https://github.com/<your-fork>/authentiguard.git
cd authentiguard

cp .env.example .env
# edit .env and replace every CHANGE_ME_* value

docker compose up -d
```

The root `docker-compose.yml` (with `docker-compose.override.yml` auto-merged) brings up Postgres, Redis, MinIO, the FastAPI backend, Celery workers, and the Next.js frontend. Use `docker compose logs -f backend` to tail a single service and `docker compose down` to stop.

For production deployments use `docker compose -f docker-compose.prod.yml up -d` instead; it adds Caddy for auto-HTTPS and wires MinIO credentials through Docker secrets (see `secrets/README.md`).

## Deployment

**Deployment is human-in-the-loop.** CI automates the safe parts and
stops at the cluster boundary.

| Step | Automated? | Where |
|------|------------|-------|
| 1. `alembic upgrade head` against the target DB | **Yes** | `.github/workflows/ci.yml` → `release-staging` / `release-production` |
| 2. Container image build + push to GHCR         | **Yes** | same workflow |
| 3. Cluster rollout (`kubectl` against EKS)      | **No — manual operator task** | [`docs/DEPLOY.md`](docs/DEPLOY.md) |

When a push to `main` triggers `release-production`, the job runs
migrations, pushes images, and writes a ready-to-paste rollout
checklist to the Actions run summary. **The job does not touch the
cluster.** An operator reads the checklist and applies it with
`kubectl`.

Wiring step 3 into CI would require AWS OIDC credentials, ECR push
instead of GHCR, Kustomize image substitution, and rollout-status
polling — none of which are in place today. Rather than fake the
automation with placeholder commands, the workflow stops honestly at
step 2 and documents step 3 as an explicit human operation. See
[`docs/DEPLOY.md`](docs/DEPLOY.md) for the full runbook, pre-rollout
checklist, smoke tests, and rollback procedure.

### Required environment variables

Copied from `.env.example`:

| Variable | Description |
|---|---|
| `APP_ENV` | Runtime environment: `development`, `staging`, or `production`. |
| `APP_DEBUG` | Enables verbose FastAPI debug output. Off in production. |
| `APP_SECRET_KEY` | App-wide secret for session/state signing. Generate with `openssl rand -hex 32`. |
| `POSTGRES_USER` | Postgres role used by the backend. |
| `POSTGRES_PASSWORD` | Password for `POSTGRES_USER`. |
| `POSTGRES_DB` | Database name. |
| `DATABASE_URL` | `postgresql+asyncpg://` connection string used by SQLAlchemy. |
| `REDIS_PASSWORD` | Password for the Redis instance. |
| `REDIS_URL` | Redis connection string for the cache (db 0). |
| `MINIO_ROOT_USER` | MinIO root account username (bootstrap/admin). |
| `MINIO_ROOT_PASSWORD` | MinIO root account password (bootstrap/admin). |
| `S3_BUCKET_UPLOADS` | Bucket for user-submitted content. |
| `S3_BUCKET_REPORTS` | Bucket for generated detection reports. |
| `S3_ENDPOINT_URL` | S3 endpoint — point at MinIO locally, omit/blank for real AWS. |
| `AWS_ACCESS_KEY_ID` | Access key used by the backend/worker S3 client. |
| `AWS_SECRET_ACCESS_KEY` | Secret key used by the backend/worker S3 client. |
| `AWS_REGION` | AWS region string. Required even for MinIO. |
| `JWT_SECRET_KEY` | HS256 signing key for access/refresh tokens. Generate with `openssl rand -hex 64`. |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | Access-token lifetime in minutes. |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | Refresh-token lifetime in days. |
| `JWT_ALGORITHM` | JWT signing algorithm (default `HS256`). |
| `CELERY_BROKER_URL` | Redis URL for the Celery broker (db 0). |
| `CELERY_RESULT_BACKEND` | Redis URL for Celery task results (db 1). |
| `FLOWER_USER` | Basic-auth username for the Flower queue dashboard. |
| `FLOWER_PASSWORD` | Basic-auth password for Flower. |
| `OPENAI_API_KEY` | OpenAI key — only used for dataset generation scripts. |
| `ANTHROPIC_API_KEY` | Anthropic key — only used for dataset generation scripts. |
| `MLFLOW_TRACKING_URI` | MLflow server URL for experiment tracking. |
| `WANDB_API_KEY` | Weights & Biases API key for training runs. |
| `ENCRYPTION_KEY` | Fernet key for at-rest field encryption. Generate with `cryptography.fernet.Fernet.generate_key()`. |
| `RATE_LIMIT_FREE_TIER` | Requests/minute for the free tier. |
| `RATE_LIMIT_PRO_TIER` | Requests/minute for the pro tier. |
| `RATE_LIMIT_ENTERPRISE_TIER` | Requests/minute for the enterprise tier. |
| `NEXT_PUBLIC_API_URL` | Backend base URL exposed to the Next.js frontend. |

### Hitting the API

- Base URL: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health probe: `GET http://localhost:8000/health`

Example — submit text for detection:

```bash
curl -X POST http://localhost:8000/api/v1/analyze/text \
  -H 'Content-Type: application/json' \
  -d '{"text": "The quick brown fox jumps over the lazy dog."}'
```

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
