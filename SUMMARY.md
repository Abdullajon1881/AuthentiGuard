# AuthentiGuard — Project Status

## Production-Ready

### Text Detection
- **4-layer ensemble**: perplexity, stylometry, DistilBERT transformer, adversarial detection
- **Accuracy**: F1 0.9555 on test set, AUROC 0.9897
- **Adversarial robustness**: 80% accuracy on hard adversarial samples (humanized AI, AI-ified human, mixed)
- **Threshold-calibrated**: optimal threshold 0.80 via validation sweep
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
- Smoke tests: health, auth flow, text analysis
- Integration tests: API flow basics
- Frontend: jest configured, minimal coverage
- No comprehensive test suite yet

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
