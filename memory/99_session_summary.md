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
