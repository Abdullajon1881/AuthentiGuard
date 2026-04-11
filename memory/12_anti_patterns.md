# Anti-Patterns — NEVER DO THESE

## Architecture
- NEVER rewrite architecture without reading ALL memory files first
- NEVER introduce a new web framework (no Flask, Django, Express — we use FastAPI)
- NEVER introduce a new task queue (no Dramatiq, Huey, RQ — we use Celery)
- NEVER introduce a new ORM (no Tortoise, Peewee — we use SQLAlchemy 2)
- NEVER replace PostgreSQL with another database
- NEVER replace Redis with another broker

## Detection Pipeline
- NEVER use a single model for any modality (ensemble is mandatory)
- NEVER remove pipeline steps: ensemble → calibrate → evidence → result
- NEVER return detection scores without layer_scores and evidence
- NEVER skip calibration (even if just passthrough until trained)
- NEVER make analysis endpoints synchronous (Celery queue is mandatory for all)

## Workers
- NEVER create a second worker for the same content type
- NEVER import detectors at module level (lazy-load via singleton)
- NEVER bypass BaseDetectionWorker (all workers must extend it)
- NEVER remove error handling (ValueError=no retry, else=retry 3x)

## Infrastructure
- NEVER use `libgl1-mesa-glx` in Dockerfiles (removed in Debian Trixie, use `libgl1`)
- NEVER pin spacy to exact version (causes typer/fastapi-cli conflicts)
- NEVER forget to list ALL queues in Dockerfile.worker CMD
- NEVER skip MinIO bucket creation on startup
- NEVER commit `.env` with real secrets to git
- NEVER expose PostgreSQL/Redis ports in production docker-compose

## Frontend
- NEVER require login for the demo (use OptionalCurrentUser on backend)
- NEVER hardcode `isLoggedIn()` gates on analysis functions
- NEVER hardcode layer count in confidence text (use dynamic count)

## Auth
- NEVER use CurrentUser on analyze/job endpoints (use OptionalCurrentUser)
- NEVER delete the demo user seed logic in main.py lifespan
- NEVER change the demo user UUID (00000000-0000-4000-a000-000000000001)
