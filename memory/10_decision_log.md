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
- 2026-04-11 | CPU-only PyTorch in worker image | Worker runs on python:3.11-slim (no CUDA); CUDA torch wastes ~1.8GB
- 2026-04-11 | Remove GPT-2 pre-download from Dockerfile | Models lazy-load via singleton pattern; model_cache volume persists across restarts
- 2026-04-11 | Split requirements into base/ml/dev layers | Worker installs only base+ml; dev tools (mlflow, wandb, dvc) excluded from production image
- 2026-04-11 | model_cache named volume at /models | Prevents GPT-2 re-download on container restart; HF_HOME/TRANSFORMERS_CACHE/TORCH_HOME all point here
- 2026-04-12 | Phase 1A: Pass checkpoint=None to TextDetector (not fake paths) | Explicit MVP mode (L1+L2 only); avoids misleading references to non-existent checkpoint dirs
- 2026-04-12 | Broaden _get_detector() except to catch Exception (was ImportError/FileNotFoundError/OSError) | GPT-2 download failure raises ConnectionError/RuntimeError not caught by narrow list; worker must never crash
- 2026-04-12 | L1 HUMAN_PPL_MEAN 120→85 (calibrated on 40 samples) | Old value too high; GPT-2 on real human text averages ~85 ppl, not 120. Biggest accuracy gain was from this single constant
- 2026-04-12 | L1 signal weights 60/30/10→45/25/30 (ppl/burst/low_frac) | Fisher discriminant shows low_ppl_frac is strongest signal (1.04) but had only 10% weight; rebalanced to match discriminant ratios
- 2026-04-12 | Adaptive label thresholds by active layer count (2→0.55, 3→0.65, 4→0.75) | 2-layer MVP scores cluster 0.20-0.68; static 0.75 threshold made AI label unreachable. Validation: 50%→80% accuracy
- 2026-04-12 | L3 uses deberta-v3-small (44M params) not deberta-v3-base (86M) | Smaller disk footprint (~180MB vs ~400MB), faster CPU inference, fits Docker <2.2GB budget
- 2026-04-12 | L3 layer_name="transformer" not "semantic" | LayerScoresSchema has `transformer` field; using "semantic" would require schema change and break evidence_summary mapping
- 2026-04-12 | Pretrained DeBERTa baked into Docker image at /models/deberta_v3_small | Avoids runtime download; on first deploy Docker copies to model_cache volume; available for fine-tuning immediately
