# Tech Stack

## Frontend
- Next.js 14 (dashboard app)
- React 18, TypeScript, Tailwind CSS
- Single-file landing page: `frontend/public/landing.html` (~3300 lines HTML/CSS/JS)

## Backend
- FastAPI (Python 3.11)
- SQLAlchemy 2 (async) + Alembic migrations
- Pydantic v2 for validation
- structlog for structured logging
- uvicorn (4 workers in production)

## Queue / Workers
- Celery 5.4 + Redis 7 (broker + result backend)
- 5 queues: text, image, audio, video, webhook
- Flower for monitoring

## Database
- PostgreSQL 16 (async via asyncpg)
- JSONB for detection evidence
- UUID primary keys

## Object Storage
- MinIO (dev) / S3 (production)
- Buckets: `ag-uploads`, `ag-reports`
- boto3 client

## AI / ML
- PyTorch 2.3
- Transformers (HuggingFace): GPT-2, DeBERTa, Wav2Vec2, CodeBERT
- timm: EfficientNet-B4, ViT-B/16, XceptionNet
- scikit-learn, XGBoost, LightGBM
- librosa (audio), opencv-headless (image/video), mediapipe (face detection)
- ffmpeg (video frame extraction)
- ONNX Runtime (inference optimization, future)

## Experiment Tracking
- MLflow 2.13
- Weights & Biases
- DVC for data versioning

## Infrastructure
- Docker Compose (dev: 8 services)
- Kubernetes (Kustomize, not Helm)
- Terraform (AWS: VPC, EKS, RDS, ElastiCache, S3)
- Caddy 2 (reverse proxy, auto-HTTPS)
- GitHub Actions (CI/CD)

## Security
- AES-256-GCM (field encryption)
- ECDSA (report signing)
- JWT HS256 (auth tokens)
- bcrypt (password hashing)
