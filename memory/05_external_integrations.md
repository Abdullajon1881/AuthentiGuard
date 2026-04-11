# External Integrations

## MinIO (S3-compatible storage)
- **Purpose:** File uploads + report storage
- **Buckets:** `ag-uploads` (user files), `ag-reports` (PDF/JSON reports)
- **Auto-created:** on FastAPI startup via `_ensure_minio_buckets()` in main.py
- **Client:** boto3 (sync) in `backend/app/services/s3_service.py`
- **Dev ports:** 9000 (API), 9001 (console), overridden to 9002/9003
- **Credentials:** `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` in .env

## Redis 7
- **Celery broker:** db 0 (`CELERY_BROKER_URL`)
- **Celery results:** db 1 (`CELERY_RESULT_BACKEND`)
- **Rate limiting:** used by `RateLimitMiddleware`
- **Requires auth:** `--requirepass` in docker-compose
- **Dev ports:** 6379, overridden to 6380

## PostgreSQL 16
- **Async driver:** asyncpg via SQLAlchemy 2
- **Tables:** users, api_keys, detection_jobs, detection_results, audit_logs, webhooks
- **Evidence:** JSONB columns (layer_scores, evidence_summary, model_attribution)
- **Init:** `Base.metadata.create_all` in lifespan (all environments)
- **Dev port:** 5432, overridden to 5434

## HuggingFace Models (pretrained)
- GPT-2: text perplexity analysis (pre-downloaded in Dockerfile.worker)
- DeBERTa-v3-base: text transformer classifier (optional, L3)
- EfficientNet-B4: image classification (via timm, pretrained=True)
- ViT-B/16: image/video vision transformer (via timm)
- XceptionNet: video face forensics (via timm)
- ResNet-18: audio classification (torchvision)
- Wav2Vec2: audio transformer (HuggingFace)
- CodeBERT: code analysis (HuggingFace)
- **None are fine-tuned** — all use pretrained weights as fallback

## MLflow
- **Purpose:** Experiment tracking
- **Port:** 5000
- **Backend store:** PostgreSQL (separate `mlflow` database)
- **Artifact store:** MinIO bucket `mlflow-artifacts`

## Flower
- **Purpose:** Celery queue monitoring
- **Port:** 5555
- **Auth:** basic auth via `FLOWER_USER` / `FLOWER_PASSWORD`

## Caddy 2
- **Purpose:** Reverse proxy + automatic HTTPS (Let's Encrypt)
- **Config:** `Caddyfile` in repo root
- **Routes:** `/api/*` → backend:8000, `/` → backend:8000 (landing), rest → frontend:3000
- **Domain:** configurable via `$DOMAIN` env var

## GitHub Actions
- **CI:** lint (ruff, mypy) + test (pytest) + Docker build check
- **CD:** GHCR push + K8s deploy (staging on develop, production on main)
- **Config:** `.github/workflows/ci.yml`

## HuggingFace Spaces
- **Purpose:** Public Gradio demo of text detector
- **Location:** `hf_space/app.py`
- **Standalone:** no backend deps, self-contained heuristic detector
