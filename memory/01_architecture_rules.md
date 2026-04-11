# Architecture Rules (NON-NEGOTIABLE)

These constraints are derived from ADR-001 through ADR-008 and battle-tested patterns.
Violating any of these will break the system or degrade detection quality.

## Structure
- Monorepo: `frontend/`, `backend/`, `ai/`, `infra/`, `security/`, `docs/`
- Directory names use **underscores** (e.g., `text_detector`, not `text-detector`)
- `ai/__init__.py` exists — `ai` is a proper Python package

## Detection Architecture
- Every modality uses a **multi-layer ensemble** (never single-model)
- Each layer is independently trained or rule-based
- XGBoost meta-classifier combines layers (weighted average fallback when untrained)
- **Platt + isotonic calibration** required for all probability outputs
- All detector outputs normalized to `DetectorOutput` via dispatcher

## Processing
- All media analysis is **async via Celery + Redis** (never synchronous HTTP)
- 5 queues: text, image, audio, video, webhook
- Workers lazy-load detectors via **singleton pattern**: `_detector = None`
- Priority levels: free=1, pro=5, enterprise=9
- Clients poll `GET /api/v1/jobs/{id}` or register webhooks

## Data & Evidence
- Every result includes `layer_scores`, `top_signals`, `evidence_summary`
- Evidence stored as **PostgreSQL JSONB** (not normalized tables)
- Results are immutable after job completion

## Security
- **JWT + refresh token rotation** (single-use refresh tokens)
- **AES-256-GCM** for application-layer field encryption
- ECDSA for report signing
- SSRF protection on URL analysis
- Rate limiting by tier (free=10, pro=100, enterprise=1000 req/min)

## Auth
- `CurrentUser` dependency for authenticated-only endpoints
- `OptionalCurrentUser` for demo/public endpoints (falls back to demo user)
- Demo user: UUID `00000000-0000-4000-a000-000000000001`

## Infrastructure
- Docker Compose for dev (8 services)
- Kubernetes + Terraform for production AWS
- Caddy for reverse proxy + auto-HTTPS
- MinIO for local S3-compatible storage
