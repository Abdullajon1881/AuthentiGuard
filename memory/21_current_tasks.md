# Current Tasks

## Completed (April 2026 Session)

- [x] Enable all 5 Celery queues in Dockerfile.worker CMD
- [x] DB `create_all` in all environments (not just dev)
- [x] MinIO bucket auto-creation on FastAPI startup
- [x] Demo user seeded on startup (UUID 00000000-0000-4000-a000-000000000001)
- [x] OptionalCurrentUser dependency on all analyze + job endpoints
- [x] Remove auth gates from landing page JS (runAnalysis, runMediaUploadAnalysis)
- [x] Layer name mappings for all 5 content types in landing.html
- [x] Poll timeout increased: 2min → 5min (maxAttempts 60 → 150)
- [x] CSP header: added Google Fonts domains
- [x] Caddyfile: fixed routing order, domain configurable via $DOMAIN
- [x] Deploy script created (deploy.sh)
- [x] CI env vars fixed (.github/workflows/ci.yml)
- [x] HuggingFace Space Gradio demo (hf_space/)
- [x] Outreach template (docs/outreach-template.md)
- [x] Fixed libgl1-mesa-glx → libgl1 in Dockerfile.worker
- [x] Loosened spacy pin: ==3.7.4 → >=3.7.4,<4.0
- [x] AI memory system created (memory/)

## Blocked

- [ ] Docker worker image build — all code fixes applied, needs `docker compose up -d --build backend worker`
  - Previous failures: libgl1 (fixed), spacy (fixed), containerd socket (transient)
  - Build takes ~10-15 min due to PyTorch + model downloads
