# AuthentiGuard — Architecture Decision Records

Architecture Decision Records (ADRs) document the reasoning behind significant
technical decisions. Each ADR is immutable once accepted — if a decision
changes, a new superseding ADR is created.

Status values: `Proposed` | `Accepted` | `Deprecated` | `Superseded by ADR-NNN`

---

## ADR-001: Four-layer ensemble for text detection

**Date:** 2024-01-15
**Status:** Accepted
**Deciders:** ML Engineering

### Context

AI text detection is an adversarial problem — any single detector can be
defeated by an attacker who knows its internals. We needed an architecture
that degrades gracefully under adversarial attack and provides explainability.

### Decision

Use a four-layer ensemble with independently trained models:
1. **Perplexity** (GPT-2 reference model) — statistical language model signal
2. **Stylometry** (rule-based) — linguistic features, no model dependency
3. **Transformer** (DeBERTa-v3-base) — deep semantic patterns
4. **Adversarial** (separate DeBERTa, trained on adversarial data) — robustness

An XGBoost meta-classifier combines all four into a 26-feature vector with
Platt + isotonic calibration for well-calibrated probabilities.

### Consequences

- **Positive:** Attacker must defeat all four layers simultaneously.
- **Positive:** Stylometry layer works with zero model files (fast startup).
- **Positive:** Layer scores are directly interpretable in the UI.
- **Negative:** Higher inference cost than a single model (mitigated by ONNX).
- **Negative:** More complex training pipeline.

---

## ADR-002: Celery + Redis for async job processing

**Date:** 2024-01-20
**Status:** Accepted
**Deciders:** Backend Engineering

### Context

Video and audio analysis can take 10–120 seconds. A synchronous HTTP request
would timeout for most clients and would block API server threads.

### Decision

Use Celery with Redis as the message broker and result backend.
Five queues with priority: `text_queue`, `image_queue`, `code_queue`,
`audio_queue`, `video_queue`. Pro/Enterprise jobs get priority 5–9;
free jobs get priority 1–3.

Clients poll `GET /api/v1/jobs/{id}` or register webhooks for completion.

### Alternatives considered

- **FastAPI BackgroundTasks:** Too simple — no retry, no priority, no distributed workers.
- **AWS SQS + Lambda:** Vendor lock-in; cold starts hurt latency SLA.
- **Kafka:** Overkill at this scale; operational complexity without benefit.

### Consequences

- **Positive:** Workers scale independently from the API.
- **Positive:** GPU workers only run on GPU nodes (cost savings).
- **Positive:** Flower provides real-time monitoring with no extra code.
- **Negative:** Requires Redis as a dependency.
- **Negative:** Webhook delivery must handle client downtime (3 retries with backoff).

---

## ADR-003: XGBoost meta-classifier over neural ensemble

**Date:** 2024-02-01
**Status:** Accepted
**Deciders:** ML Engineering

### Context

We needed a meta-classifier to combine detector outputs. Options were: neural
network, XGBoost, logistic regression, or simple weighted average.

### Decision

XGBoost with Platt + isotonic calibration for the ensemble meta-classifier
(both text ensemble and multi-modal ensemble in Phase 10).

### Alternatives considered

- **Neural MLP:** Requires more data; harder to interpret; overkill for 26 features.
- **Logistic regression:** Fast and interpretable but cannot model feature interactions.
- **Simple weighted average:** Loses information from feature correlations.
- **XGBoost:** Fast, handles missing features (detector failures), feature importance
  available, well-calibrated with Platt scaling, < 1ms inference.

### Consequences

- **Positive:** Feature importance explains which layer drove the decision.
- **Positive:** Gracefully handles missing detector outputs (fills 0.5 for absent layers).
- **Positive:** Inference is < 1ms (negligible overhead).
- **Negative:** Requires calibration step after training.

---

## ADR-004: JWT with single-use refresh token rotation

**Date:** 2024-02-10
**Status:** Accepted
**Deciders:** Security

### Context

Standard JWT implementations store refresh tokens without invalidation,
creating a security risk if a token is stolen.

### Decision

Refresh tokens are stored in Redis with a unique JTI (JWT ID). On each use:
1. The submitted JTI is looked up in Redis — if missing, reject (already used).
2. A new access + refresh token pair is issued.
3. The old JTI is deleted from Redis.

This implements "refresh token rotation" — a stolen refresh token can only
be used once before it's invalidated by the legitimate user's next request.

### Consequences

- **Positive:** Stolen refresh tokens have a one-use window before detection.
- **Positive:** All refresh tokens can be revoked instantly (delete all Redis keys).
- **Negative:** Concurrent requests with the same refresh token will fail
  (the second request arrives after the first has already rotated it).
- **Mitigation:** Client-side retry logic handles the rare concurrent refresh case.

---

## ADR-005: PostgreSQL JSONB for evidence storage

**Date:** 2024-02-15
**Status:** Accepted
**Deciders:** Backend Engineering

### Context

Evidence summaries (sentence scores, layer outputs, signals) have a flexible
schema that varies by content type. Options: normalised relational tables,
JSONB columns, or a document database.

### Decision

Store `evidence_summary`, `layer_scores`, and `sentence_scores` as JSONB
columns in the `DetectionResult` table. The core fields (score, label,
confidence) remain as typed columns with indexes.

### Alternatives considered

- **Normalised tables:** Schema migrations required for every new detector/signal.
- **MongoDB:** Second database to operate; joins become application-level.
- **JSONB:** Best of both — SQL joins, typed indexes on core fields, flexible schema
  for evidence. GIN indexes on JSONB for filtering by signal type.

### Consequences

- **Positive:** Schema evolves without migrations for evidence fields.
- **Positive:** GIN index enables `WHERE evidence_summary @> '{"label": "AI"}'` queries.
- **Negative:** JSONB cannot be indexed as efficiently as typed columns for range queries.
- **Mitigation:** Only query by evidence fields in analytics; core fields (score, label)
  are typed columns with B-tree indexes.

---

## ADR-006: ONNX Runtime for production inference

**Date:** 2024-03-01
**Status:** Accepted
**Deciders:** ML Engineering, Platform

### Context

PyTorch models in production have high memory usage and slower inference than
optimised runtimes. We needed < 200ms text analysis p50 latency.

### Decision

Export all trained models to ONNX format and serve via ONNX Runtime with:
- **Graph optimisations** (constant folding, operator fusion)
- **INT8 quantisation** for CPU inference (DeBERTa: 4× faster, 75% size reduction)
- **CUDAExecutionProvider** for GPU workers
- **CPUExecutionProvider** for CPU workers

### Consequences

- **Positive:** DeBERTa inference: ~250ms → ~60ms (CPU, INT8).
- **Positive:** Memory: 440MB → 110MB per model instance.
- **Positive:** No PyTorch required in the inference container (smaller image).
- **Negative:** ONNX export required for every model change.
- **Negative:** Custom PyTorch ops must be implemented in ONNX-compatible form.
- **Mitigation:** `performance/onnx/ort_export.py` automates export + validation.

---

## ADR-007: AES-256-GCM for application-layer field encryption

**Date:** 2024-03-10
**Status:** Accepted
**Deciders:** Security

### Context

Email addresses and API keys in PostgreSQL needed encryption at the application
layer (beyond database-level encryption) to protect against database breaches
where the encryption key is not stored with the data.

### Decision

Use AES-256-GCM (via Python's `cryptography` library) for sensitive DB fields.
Each ciphertext includes a random 12-byte nonce, so the same plaintext produces
different ciphertexts on each encryption. The 16-byte GCM authentication tag
prevents silent tampering.

### Alternatives considered

- **Fernet (AES-128-CBC + HMAC):** Simpler but uses CBC mode (padding oracle risk)
  and only 128-bit keys. GCM is preferred for AEAD.
- **AWS KMS envelope encryption:** Stronger key management but adds network latency
  and cost per field access. Reserved for the S3 bucket KMS keys.
- **bcrypt (for passwords):** Already used — bcrypt is one-way, not encryptable.

### Consequences

- **Positive:** AES-256-GCM provides authenticated encryption — tampering detected.
- **Positive:** Random nonce prevents deterministic ciphertext (no frequency analysis).
- **Negative:** Encrypted fields cannot be used in SQL `WHERE` or `ORDER BY` clauses.
- **Mitigation:** Only encrypt fields never queried directly (email as login uses
  a separate lookup by hashed email, not the ciphertext).

---

## ADR-008: Kustomize for environment configuration over Helm templating

**Date:** 2024-03-20
**Status:** Accepted
**Deciders:** Platform Engineering

### Context

We needed environment-specific Kubernetes configuration (staging vs production)
without duplicating manifests. Options: Helm templates, Kustomize overlays, or
plain YAML with sed substitution.

### Decision

Use Kustomize for base + overlay configuration. Helm is used as a packaging
and dependency mechanism (for third-party charts), but our first-party manifests
use Kustomize overlays.

### Rationale

Kustomize is built into `kubectl` (no extra tool), treats YAML as data rather
than templates, and its patch semantics are explicit and auditable. Helm
templating with `{{ if eq .Values.env "production" }}` blocks becomes hard
to read and test.

The combination — Kustomize overlays deployed via ArgoCD — provides GitOps
with full auditability.

### Consequences

- **Positive:** Base manifests are valid YAML with no template syntax.
- **Positive:** Diffs between environments are visible as patches, not template logic.
- **Positive:** `kustomize build | kubectl diff` shows exactly what would change.
- **Negative:** Kustomize patches can be verbose for large structural changes.
- **Mitigation:** Helm values.yaml handles complex parameterisation; Kustomize
  handles environment-specific overrides.
