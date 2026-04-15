# AuthentiGuard — Project Status

## Production-Ready

### Text Detection
- **3-layer ensemble** at inference: L1 perplexity (GPT-2) + L2 stylometry (spaCy) + L3 DeBERTa-v3-small (44M params, fine-tuned on an adversarial-augmented corpus). L4 adversarial and the XGBoost meta-classifier exist in code but are **not loaded** in production (`adversarial_checkpoint=None`, `meta_checkpoint=None` in `backend/app/workers/text_worker.py`).
- **L3 alone, validation split, training-time eval (upward-biased because selection and reporting share a dataset)**: F1 **0.9498**, AUROC **0.9818**. Source: `ai/text_detector/checkpoints/transformer_v3_hard/phase1/checkpoint-3582/trainer_state.json`.
- **L1+L2+L3 ensemble, held-out v1 test split, post-fit (AUTHORITATIVE)**: F1 **0.9945**, precision **0.9960**, recall **0.9930**, AUROC **0.9977**. Source: `ai/text_detector/accuracy/ensemble_test_eval.post_fit.json`. Confusion matrix `[[1000, 4], [7, 989]]` on 2000 rows.
- **L1+L2+L3 ensemble, held-out v2 test split, post-fit (includes adversarial subsets, AUTHORITATIVE)**: F1 **0.9529**, precision 0.9243, recall **0.9832**, AUROC 0.9767. Source: `ai/text_detector/accuracy/ensemble_test_eval_v2.post_fit.json`. 3482 rows.
- **Adversarial subset (`adv_mixed` in v2 test, n=169)**: F1 pre-fit **0.606** → post-fit **0.846** (+24 points from fitting weights and threshold on val).
- **Combiner weights and AI threshold** are fit on val data via grid search (`scripts/fit_ensemble_weights.py`). Current values: weights `[0.20, 0.35, 0.45]` (L1/L2/L3), AI threshold `0.41`. Val F1 at these values: 0.9969. Val-test F1 gap: 0.00245. Source: `ai/text_detector/accuracy/fit_weights.json`.
- **Full audit trail:** [`ai/text_detector/ACCURACY.md`](ai/text_detector/ACCURACY.md) — every number above is traceable to a persisted JSON artifact with git SHA and dataset SHA-256.

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
