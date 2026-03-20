# AuthentiGuard — Developer Onboarding Guide

Welcome to the AuthentiGuard codebase. This guide gets you from zero to a
running local environment and your first successful detection call in under
30 minutes.

---

## Prerequisites

| Tool           | Version  | Install                                  |
|----------------|----------|------------------------------------------|
| Python         | ≥ 3.11   | [python.org](https://python.org)         |
| Node.js        | ≥ 20     | [nodejs.org](https://nodejs.org)         |
| Docker Desktop | ≥ 25     | [docker.com](https://docker.com)         |
| Git            | ≥ 2.40   | `brew install git` / `apt install git`   |
| Make           | any      | pre-installed on macOS/Linux             |

GPU support (optional, speeds up video/audio detection):
- NVIDIA GPU with CUDA 12+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## 1. Clone and configure

```bash
git clone https://github.com/authentiguard/authentiguard.git
cd authentiguard

# Copy environment template
cp .env.example .env

# Generate secure secrets (macOS/Linux)
python3 -c "
import secrets, base64, os
keys = {
    'JWT_SECRET_KEY':    secrets.token_hex(64),
    'APP_SECRET_KEY':    secrets.token_hex(32),
    'ENCRYPTION_KEY':    base64.b64encode(os.urandom(32)).decode(),
}
for k, v in keys.items():
    print(f'{k}={v}')
"
# Paste the output into your .env file
```

Your `.env` must contain at minimum:

```bash
DATABASE_URL=postgresql+asyncpg://authentiguard:authentiguard@localhost:5432/authentiguard
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=<generated above>
APP_SECRET_KEY=<generated above>
ENCRYPTION_KEY=<generated above>
AWS_ACCESS_KEY_ID=minioadmin        # MinIO for local dev
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_REGION=us-east-1
S3_ENDPOINT_URL=http://localhost:9000
S3_BUCKET_UPLOADS=authentiguard-uploads
S3_BUCKET_REPORTS=authentiguard-reports
```

---

## 2. Start the full stack

```bash
# Pull images and start all services
docker compose up -d

# Wait for services to be healthy (30–60s)
docker compose ps

# Expected output: all services "healthy" or "running"
# postgres, redis, minio, api, worker, flower, frontend, mlflow
```

Check everything is running:

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "0.1.0", "uptime_s": 12.3}

curl http://localhost:3000
# Next.js frontend (open in browser)

curl http://localhost:5555
# Flower — Celery task monitor

curl http://localhost:5000
# MLflow experiment tracking
```

---

## 3. Install Python development dependencies

```bash
# Backend
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# AI modules
cd ../ai
pip install -r requirements.txt

# Return to root
cd ..
```

---

## 4. Run database migrations

```bash
cd backend
alembic upgrade head

# Verify tables created
docker compose exec postgres psql -U authentiguard -d authentiguard -c "\dt"
```

---

## 5. Create a test user and make your first detection call

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "dev@local.test",
    "password": "Dev@Local123!",
    "full_name": "Local Dev",
    "consent_given": true
  }'

# Login
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "dev@local.test", "password": "Dev@Local123!"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Token: ${TOKEN:0:40}..."

# Detect AI text
curl -X POST http://localhost:8000/api/v1/analyze/text \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Furthermore, it is worth noting that this approach leverages robust mechanisms to facilitate comprehensive understanding of nuanced concepts. Additionally, the multifaceted paradigm enables sophisticated optimization."
  }'
```

Expected response:
```json
{
  "job_id": "...",
  "content_type": "text",
  "authenticity_score": 0.83,
  "label": "AI",
  "confidence": 0.66,
  "confidence_level": "medium",
  "processing_ms": 340,
  "layer_scores": { "perplexity": 0.78, "stylometry": 0.71, "transformer": 0.88, "adversarial": 0.83 }
}
```

---

## 6. Run the test suite

```bash
# Fast unit tests (no I/O, < 30s)
pytest -m unit -q

# Backend unit tests
cd backend && pytest tests/unit/ -q

# AI unit tests
cd ai && pytest -q

# Security tests
cd ../security && pytest -q

# Full test suite (requires running Docker stack)
pytest -m "not e2e and not slow" --tb=short

# With coverage report
pytest --cov=backend --cov=ai --cov-report=html
open reports/coverage/index.html
```

---

## 7. Project structure

```
authentiguard/
├── backend/               FastAPI API + Celery workers
│   ├── app/
│   │   ├── api/v1/        Route handlers
│   │   ├── core/          Config, DB, Redis, JWT
│   │   ├── models/        SQLAlchemy ORM models
│   │   ├── schemas/       Pydantic request/response schemas
│   │   ├── services/      Upload, metadata, report services
│   │   ├── middleware/     Rate limiting, audit logging
│   │   └── workers/       Celery task definitions
│   └── tests/
├── frontend/              Next.js 14 + React + TypeScript
│   └── src/
│       ├── app/           Next.js app router pages
│       ├── components/    UI components (upload, analysis, dashboard)
│       ├── hooks/         Custom React hooks
│       ├── lib/           API client, types
│       └── tests/
├── ai/                    All ML detection modules
│   ├── text-detector/     4-layer text ensemble
│   ├── audio-detector/    Audio deepfake detection
│   ├── video-detector/    Video deepfake detection
│   ├── image-detector/    GAN/diffusion image detection
│   ├── code-detector/     AST + transformer code detection
│   ├── ensemble-engine/   Multi-modal meta-classifier
│   └── authenticity-engine/ Unified scoring + reports
├── security/              Encryption, GDPR, SOC 2 controls
├── performance/           Caching, batching, benchmarks
├── infra/                 Kubernetes, Helm, Terraform
│   ├── k8s/
│   │   ├── base/          Base Kubernetes manifests
│   │   ├── overlays/      Kustomize environment overlays
│   │   └── helm/          Helm chart
│   └── terraform/         AWS infrastructure
└── docs/                  This documentation
    ├── api/               OpenAPI spec
    ├── architecture/      ADRs, system diagrams
    ├── onboarding/        This guide
    └── guides/            Runbooks and how-to guides
```

---

## 8. Development workflow

### Making a change

```bash
# Create a feature branch
git checkout -b feature/my-feature develop

# Make changes, then run relevant tests
pytest backend/tests/unit/ -q

# Run the linter
cd backend && ruff check . && mypy app/

# Commit with conventional commit message
git commit -m "feat(text-detector): improve stylometry scoring for short texts"
```

### Commit message format

```
<type>(<scope>): <short summary>

Types: feat | fix | docs | refactor | test | chore | perf | security
Scope: text-detector | audio | video | image | code | backend | frontend | infra

Examples:
  feat(audio): add pitch jitter analysis to Phase 3 adversarial training
  fix(backend): correct JWT expiry calculation for timezone-aware datetimes
  docs(onboarding): add GPU setup instructions for Apple Silicon
  perf(caching): reduce MinHash computation time by 40% with vectorised ops
```

### Adding a new detector layer

1. Create `ai/text-detector/layers/layer_N_yourname.py` implementing `BaseDetectionLayer`.
2. Add to `ai/text-detector/ensemble/text_detector.py` with a weight.
3. Extend `FEATURE_NAMES` in the meta-classifier to include your layer's features.
4. Add unit tests in `ai/text-detector/tests/`.
5. Update `docs/architecture/adr.md` with an ADR if the change is significant.

---

## 9. Useful commands reference

```bash
# View Celery worker logs
docker compose logs -f worker

# Inspect Celery queue depths
docker compose exec redis redis-cli llen text_queue

# Connect to PostgreSQL
docker compose exec postgres psql -U authentiguard -d authentiguard

# Run database migrations
cd backend && alembic upgrade head

# Revert last migration
cd backend && alembic downgrade -1

# Generate a new migration
cd backend && alembic revision --autogenerate -m "add index on jobs.created_at"

# Rebuild a single service after code change
docker compose up -d --build api

# Access MinIO console (local S3)
open http://localhost:9001   # user: minioadmin / minioadmin

# Tail all service logs
docker compose logs -f

# Stop everything
docker compose down

# Stop and delete volumes (full reset)
docker compose down -v
```

---

## 10. Getting help

- **Slack:** `#authentiguard-eng` — engineering questions
- **Slack:** `#authentiguard-incidents` — production alerts
- **GitHub Discussions:** Architecture questions and RFCs
- **Issues:** Bug reports and feature requests

When asking for help, include:
1. The exact command you ran
2. The full error message (not just the last line)
3. Your OS and Python version (`python --version`)
4. Whether the Docker stack is running (`docker compose ps`)
