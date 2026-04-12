# Session Summaries

## Session: 2026-04-10

### Built
- OptionalCurrentUser auth dependency for anonymous demo access
- Demo user seeding in FastAPI lifespan
- MinIO bucket auto-creation on startup
- DB table creation in all environments
- Deploy script (deploy.sh)
- HuggingFace Space Gradio demo (hf_space/)
- Outreach template (docs/outreach-template.md)
- AI memory system (memory/ — 16 files)

### Changed
- Dockerfile.worker: all 5 queues, libgl1 fix
- routes.py: 6 endpoints from CurrentUser → OptionalCurrentUser
- landing.html: removed auth gates, added layer mappings, 5min poll timeout
- Caddyfile: fixed routing, configurable domain
- ai/requirements.txt: loosened spacy pin
- ci.yml: added missing env vars

### Decisions
- OptionalCurrentUser over duplicate endpoints for anonymous access
- Demo user UUID fixed at 00000000-0000-4000-a000-000000000001
- Hetzner CX51 recommended for hosting (~30 EUR/month)
- Loosened spacy pin to avoid typer conflict
- Memory system uses tiered structure (Tier 0-3 + Meta)

### Risks Discovered
- Docker containerd socket drops during large image export
- All detectors use pretrained weights — accuracy low for image/audio/video
- Demo user jobs accumulate without cleanup cron
- CSP header may need updates if landing page adds external resources

### Next Step
- Run `docker compose up -d --build backend worker` and verify

## Session: 2026-04-11

### Built
- 3-stage Dockerfile.worker (base → deps → runtime)
- Split requirements: `requirements/base.txt`, `ml.txt`, `dev.txt`
- `.dockerignore` excluding .git, docs, tests, datasets, model files, memory
- `model_cache` named volume for persistent HF model cache

### Changed
- `Dockerfile.worker`: complete rewrite — 3 stages, CPU-only PyTorch, no GPT-2 pre-download
- `docker-compose.yml`: added `model_cache:` volume + mount on worker
- `docker-compose.prod.yml`: same volume additions

### Decisions
- CPU-only PyTorch (CUDA torch wasted ~1.8GB in slim image with no GPU)
- GPT-2 lazy-loads at runtime instead of baking into image
- model_cache volume persists models across container restarts
- requirements split: worker installs only base+ml, not dev tools

### Risks Discovered
- First text detection on fresh deploy has 5-10 min cold start (GPT-2 download)

### Next Step
- Run `docker compose build worker` to verify build and image size

## Session: 2026-04-12

### Built
- 4-phase accuracy roadmap (text F1 0.60→0.90, image 0.50→0.80, audio 0.50→0.85)

### Analysis Completed
- Full code review of all detector layers (L1-L4 text, image ensemble, audio ensemble)
- Identified _DevFallbackDetector as actual running text detector (not the real TextDetector)
- Confirmed all training infrastructure exists but was never executed
- Mapped exact checkpoint paths and loading code for each detector

### Decisions
- Phase 1 priority: activate real TextDetector (L1+L2 MVP mode) before any training
- DeBERTa-v3-base (not -large) for Phase 2 — fits CPU worker, ~300ms inference
- HC3 + RAID + OpenGPTText for text training data (~50K samples)
- CIFAKE + GenImage subset for image training (~100K samples)
- All model checkpoints go in model_cache volume, not Docker image

### Next Step
- Execute Phase 1A: fix `_get_detector()` in `text_worker.py` to import and use real TextDetector
