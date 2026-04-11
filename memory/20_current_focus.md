# Current Focus

**Phase:** Production launch
**Date:** 2026-04-10

## Status
- All 8 implementation phases COMPLETED in code
- Docker worker build BLOCKED (deps fixed, needs rebuild)

## What Was Done This Session
1. Enabled all 5 Celery queues in Dockerfile.worker
2. DB tables created in all environments (not just dev)
3. MinIO bucket auto-creation on startup
4. Demo user seeded on startup for anonymous access
5. OptionalCurrentUser on all analyze + job endpoints
6. Removed auth gates from landing page JS
7. Layer name mappings for all 5 content types
8. Poll timeout: 2min → 5min
9. CSP header: added Google Fonts
10. Caddyfile: fixed routing, made domain configurable
11. Deploy script created
12. CI env vars fixed
13. HF Space Gradio demo created
14. Outreach template created
15. Fixed libgl1-mesa-glx → libgl1 (Debian Trixie)
16. Fixed spacy pin (loosened to >=3.7.4,<4.0)

## Blocker
- Docker `docker compose up --build backend worker` needs to run
- Previous builds failed due to libgl1 and spacy issues (both fixed now)
- Build takes ~10-15 min

## Next Action
1. Run `docker compose up -d --build backend worker`
2. Verify landing page at http://localhost:8000
3. Test text detection without login
4. Test file uploads for image/audio/video
