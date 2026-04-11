# Decision Log

Format: `DATE | DECISION | REASON`

## Architecture Decisions (from ADRs)

- 2024-01-15 | Four-layer ensemble for text detection (perplexity + stylometry + DeBERTa + adversarial) | Attacker must defeat all layers; single-model too fragile
- 2024-01-20 | Celery + Redis for async processing (5 queues with priority) | Video/audio take 10-120s; sync HTTP would timeout
- 2024-02-01 | XGBoost meta-classifier over neural ensemble | 26 features, interpretable, models feature interactions, fast
- 2024-02-10 | JWT with single-use refresh token rotation | Stolen refresh token can only be used once; rotation detects theft
- 2024-02-15 | PostgreSQL JSONB for evidence storage | Flexible schema for varying detector outputs; avoid table-per-modality
- 2024-03-01 | ONNX Runtime for production inference | 2-4x speedup over PyTorch; quantization-friendly
- 2024-03-10 | AES-256-GCM for field encryption | At-rest encryption for PII; meets compliance requirements
- 2024-03-15 | Kustomize over Helm for K8s config | Less templating complexity; overlays match our env structure

## Session Decisions (April 2026)

- 2026-04-08 | Design doc Approach B: Full Platform (all 5 modalities) | User wants complete product, not MVP subset
- 2026-04-08 | Free/open source, payment later | Mission-driven: truth and authenticity first
- 2026-04-08 | Accepted cherry-picks: HF Space, CI pipeline, outreach template | CEO review: high value, low effort
- 2026-04-08 | Deferred: accuracy benchmark dashboard | Requires trained models; no honest benchmarks possible yet
- 2026-04-10 | OptionalCurrentUser for demo endpoints | Landing page promised "no account required" but API required JWT
- 2026-04-10 | Demo user seeded on startup (UUID 00000000-0000-4000-a000-000000000001) | Anonymous requests need a user_id for job ownership
- 2026-04-10 | All 5 Celery queues in Dockerfile.worker CMD | Was only text+webhook; image/audio/video jobs silently dropped
- 2026-04-10 | DB create_all in all environments (not just dev) | First deploy needs tables; Alembic for future migrations
- 2026-04-10 | MinIO bucket auto-creation in FastAPI lifespan | File uploads fail with NoSuchBucket without it
- 2026-04-10 | Loosened spacy pin from ==3.7.4 to >=3.7.4,<4.0 | Exact pin conflicted with fastapi-cli typer dependency
- 2026-04-10 | Replaced libgl1-mesa-glx with libgl1 in Dockerfile | Package removed in Debian Trixie (python:3.11-slim)
- 2026-04-10 | CSP header allows Google Fonts (fonts.googleapis.com, fonts.gstatic.com) | Landing page loads Google Fonts; production CSP blocked them
- 2026-04-10 | Hetzner CX51 recommended for hosting | 8 vCPU, 32GB RAM, ~30EUR/month; CPU-only sufficient for launch
- 2026-04-10 | Poll timeout increased from 2min to 5min | Video processing can take 300s; old timeout too short
