# Repository Structure

```
authentiguard/
├── frontend/
│   ├── public/
│   │   └── landing.html          # Single-file landing page (~3300 lines)
│   ├── src/                      # Next.js 14 dashboard app
│   ├── Dockerfile
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI entry, lifespan (DB init, MinIO, demo user)
│   │   ├── api/v1/
│   │   │   ├── deps.py           # CurrentUser, OptionalCurrentUser
│   │   │   └── endpoints/
│   │   │       └── routes.py     # All 20 API endpoints (~800 lines)
│   │   ├── core/
│   │   │   ├── config.py         # All env var definitions (Pydantic Settings)
│   │   │   ├── database.py       # SQLAlchemy engine + AsyncSessionLocal
│   │   │   ├── security.py       # JWT, passwords, token rotation
│   │   │   └── redis.py          # Redis connection
│   │   ├── models/
│   │   │   ├── models.py         # User, DetectionJob, DetectionResult, AuditLog
│   │   │   └── webhook.py        # Webhook model
│   │   ├── schemas/
│   │   │   └── schemas.py        # Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── s3_service.py     # Shared S3/MinIO helper
│   │   │   ├── upload_service.py # File upload + type detection
│   │   │   ├── url_analyzer.py   # SSRF-protected URL fetch
│   │   │   └── report_service.py # PDF report generation
│   │   ├── workers/
│   │   │   ├── celery_app.py     # Queue config, routing, priorities
│   │   │   ├── base_worker.py    # BaseDetectionWorker (template method)
│   │   │   ├── text_worker.py    # Text detection + _DevFallbackDetector
│   │   │   ├── image_worker.py   # Image detection worker
│   │   │   ├── audio_worker.py   # Audio detection worker
│   │   │   ├── video_worker.py   # Video detection worker
│   │   │   ├── webhook_worker.py # Webhook delivery
│   │   │   └── cleanup.py        # Stuck job cleanup (periodic)
│   │   └── middleware/
│   │       └── middleware.py      # Rate limiting, audit logging
│   ├── Dockerfile
│   └── requirements.txt
│
├── ai/
│   ├── __init__.py               # Makes ai a Python package
│   ├── text_detector/
│   │   └── ensemble/
│   │       └── text_detector.py  # 4-layer: perplexity, stylometry, DeBERTa, adversarial
│   ├── image_detector/
│   │   ├── image_detector.py     # EfficientNet + ViT + hand-crafted features
│   │   ├── models/classifier.py  # Ensemble classifier (pretrained fallback)
│   │   ├── features/extractor.py # GAN fingerprint, FFT, texture
│   │   └── pipeline/             # Preprocessing
│   ├── audio_detector/
│   │   ├── audio_detector.py     # CNN + ResNet + Wav2Vec2, chunk-based
│   │   ├── models/classifier.py  # Audio ensemble
│   │   └── features/extractor.py # Spectral features
│   ├── video_detector/
│   │   ├── video_detector.py     # XceptionNet + EfficientNet + ViT + temporal
│   │   ├── models/classifier.py  # Video ensemble
│   │   ├── pipeline/             # Frame extraction, face detection
│   │   └── features/             # Artifact + temporal analyzers
│   ├── code_detector/
│   │   └── code_detector.py      # CodeBERT + AST analysis
│   ├── ensemble_engine/
│   │   └── routing/
│   │       └── dispatcher.py     # DetectorRegistry, lazy-load, normalize output
│   ├── authenticity_engine/
│   │   └── engine.py             # Unified scoring, C2PA, HMAC-signed reports
│   ├── Dockerfile
│   └── requirements.txt
│
├── datasets/                     # DVC-versioned training data
├── infra/
│   ├── docker/
│   │   └── postgres-init.sql
│   ├── k8s/                      # Kubernetes manifests (Kustomize)
│   └── terraform/                # AWS infrastructure (VPC, EKS, RDS)
├── security/                     # Encryption, compliance
├── docs/
│   ├── api/openapi.yaml
│   ├── architecture/adr.md       # 8 ADRs
│   ├── guides/runbooks.md        # 6 operational runbooks
│   └── onboarding/getting-started.md
├── hf_space/                     # HuggingFace Spaces Gradio demo
├── memory/                       # THIS DIRECTORY — AI memory system
│
├── docker-compose.yml            # 8 services (dev)
├── docker-compose.prod.yml       # Production overrides
├── docker-compose.override.yml   # Local port overrides
├── Dockerfile.worker             # Worker image (PyTorch + all ML deps)
├── Caddyfile                     # Reverse proxy config
├── deploy.sh                     # Production deploy script
├── .env                          # Secrets (never commit real values)
├── .env.example                  # Template
└── .env.production.example       # Production template
```
