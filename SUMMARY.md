# AuthentiGuard — Project Status

## Production-Ready

### Text Detection
- **4-layer ensemble**: perplexity, stylometry, DistilBERT transformer, adversarial detection
- **Accuracy**: F1 0.9498, AUROC 0.9818 (v3_hard adversarial checkpoint, epoch 2)
- **Non-adversarial baseline**: F1 0.9457, AUROC 0.9897 (v3 checkpoint, epoch 3 — not used in production)
- **Adversarial robustness**: trained on hard adversarial samples (humanized AI, AI-ified human, mixed)
- **Threshold-calibrated**: adaptive thresholds by active layer count (0.75 AI threshold at 4 layers)
- **Training pipeline**: dataset v2 (34k samples, 5 sources), sample weighting, hard example mining

### Backend API
- Auth: register, login, JWT refresh rotation, logout, password reset
- Analysis: text paste, file upload, URL fetch (SSRF-protected)
- Jobs: async polling, results, report export (PDF + JSON)
- Webhooks: full CRUD + HMAC-signed delivery
- Rate limiting by tier (configurable via settings)

### Infrastructure
- Docker Compose (8 services), Kubernetes manifests, Terraform (AWS)
- Alembic migrations, DVC data pipeline
- CI/CD: GitHub Actions

## Experimental / Beta

### Other Modalities
- **Image**: EfficientNet-B4 + ViT-B/16 + GAN fingerprint + FFT (model exists, not production-tested)
- **Audio**: CNN + ResNet-18 + Wav2Vec2 (model exists, not production-tested)
- **Video**: XceptionNet + temporal consistency (training script exists, no production model)
- **Code**: AST analysis + CodeBERT (model exists, not production-tested)

### Not Production-Ready
- Passport/provenance feature (disabled)
- Browser extension (non-functional placeholder)
- Non-text Celery workers (implemented but untested in production)

## Testing
- Unit tests: config, security, rate limiting, CORS, schemas, workers
- Integration tests: register/login/submit flow (mocked Celery)
- E2E tests: full submit -> queue -> worker -> result (real stack, no mocks)
- Load tests: 50-concurrent stress test with latency/error-rate thresholds
- Frontend: jest configured, minimal coverage

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11, SQLAlchemy 2, Alembic |
| Queue | Celery + Redis 7 |
| Database | PostgreSQL 16 |
| Storage | S3 / MinIO |
| ML | PyTorch, Transformers, scikit-learn |
| Infra | Docker, Kubernetes, Terraform |
