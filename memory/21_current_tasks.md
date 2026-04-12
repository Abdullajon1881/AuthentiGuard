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

## Completed (April 11 Session)

- [x] Docker worker image optimization (3-stage build, CPU-only PyTorch)
- [x] Split requirements into base.txt / ml.txt / dev.txt
- [x] Created .dockerignore
- [x] Added model_cache volume for persistent HF model cache
- [x] Removed GPT-2 pre-download from Dockerfile (lazy-loads at runtime)

## Completed (April 12 Session)

- [x] Phase 1A: Activate real TextDetector (L1+L2) in text_worker.py
  - Changed `_get_detector()` to pass `transformer_checkpoint=None, device="cpu"`
  - Broadened exception handling from 3 types to `Exception`
  - _DevFallbackDetector preserved as fallback
- [x] Install spaCy en_core_web_sm in Dockerfile.worker for full L2 stylometry
  - Direct pip install of whl from spacy-models GitHub releases
  - Installed in deps stage with `--prefix=/install`, carried to runtime via COPY
- [x] Phase 1B: Calibrate L1 perplexity constants from empirical data
  - Ran 40-sample calibration (20 human + 20 AI) through GPT-2
  - Updated HUMAN_PPL_MEAN 120→85, LOW_PPL_THRESHOLD 50→42, weights 60/30/10→45/25/30
  - L1 accuracy improved 65%→75% on calibration set
- [x] Phase 1 validation: end-to-end test of L1+L2 pipeline (20 samples)
  - Before adaptive thresholds: 10/20 correct (50%), 0/10 AI detected
  - After adaptive thresholds: 16/20 correct (80%), 6/10 AI detected
  - Human: 6/10 HUMAN + 4 UNCERTAIN, 0 false positives
  - AI: 6/10 AI + 3 UNCERTAIN, 1 false negative (claude_professional)
  - Schema valid, ~100ms/sample on CPU
- [x] Adaptive label thresholds based on active layer count
  - 2 layers: AI≥0.55, HUMAN<0.30
  - 3 layers: AI≥0.65, HUMAN<0.35
  - 4 layers: AI≥0.75, HUMAN<0.40 (unchanged from original)
- [x] L3 SemanticLayer created (ai/text_detector/layers/layer3_semantic.py)
  - DeBERTa-v3-small, CPU-only, sliding window, per-sentence scoring
  - layer_name="transformer" for schema compatibility
  - Wired into TextDetector (replaces TransformerLayer import)
  - Inactive until fine-tuned checkpoint provided (same conditional logic)
- [x] DeBERTa-v3-small pretrained model added to Dockerfile.worker (~180MB)
  - Downloaded via huggingface snapshot_download in deps stage
  - COPY to /models/deberta_v3_small in runtime stage
  - Ready for fine-tuning, NOT for inference (random classification head)

## Blocked

- [ ] Docker worker build verification — `docker compose build worker` not yet run
  - All code/config changes applied, needs build + smoke test
