# Protected Files

These files are load-bearing. Do NOT delete, rename, or significantly restructure
without reading the full file and understanding all callers.

## Backend Core
- `backend/app/main.py` — FastAPI entry, lifespan hooks (DB init, MinIO buckets, demo user seed), all middleware, route mounting
- `backend/app/core/config.py` — All env var definitions via Pydantic Settings. Adding/removing vars here breaks startup
- `backend/app/core/database.py` — SQLAlchemy engine, `AsyncSessionLocal`, `Base` class. All models inherit from Base
- `backend/app/core/security.py` — JWT creation/decode, password hashing (bcrypt), refresh token rotation, password reset tokens

## API Layer
- `backend/app/api/v1/deps.py` — `CurrentUser` + `OptionalCurrentUser` dependencies. All endpoints use these
- `backend/app/api/v1/endpoints/routes.py` — All 20 API endpoints (~800 lines). The single source of truth for API behavior
- `backend/app/schemas/schemas.py` — Pydantic request/response schemas. Breaking these breaks the API contract

## Data Models
- `backend/app/models/models.py` — ORM: User, DetectionJob, DetectionResult, AuditLog enums. Changing columns requires Alembic migration
- `backend/app/models/webhook.py` — Webhook model

## Workers
- `backend/app/workers/celery_app.py` — Queue definitions, routing map, priority config, periodic tasks
- `backend/app/workers/base_worker.py` — `BaseDetectionWorker` template. All 5 workers inherit from this
- `backend/app/workers/text_worker.py` — Contains `_DevFallbackDetector` (the only working heuristic detector)

## AI
- `ai/ensemble_engine/routing/dispatcher.py` — `DetectorRegistry`, output normalization. All detectors register here
- `ai/*/models/classifier.py` — Checkpoint loading logic with pretrained fallback. Each checks `if checkpoint_dir.exists()`

## Infrastructure
- `docker-compose.yml` — 8-service definition (postgres, redis, minio, backend, worker, flower, frontend, mlflow)
- `Dockerfile.worker` — Worker image. Must list ALL queues in CMD. Must use `libgl1` not `libgl1-mesa-glx`
- `.env` — Live secrets. NEVER commit real values. NEVER delete

## Frontend
- `frontend/public/landing.html` — The entire landing page (~3300 lines). Contains demo UI, auth modal, result display, file upload, animations
