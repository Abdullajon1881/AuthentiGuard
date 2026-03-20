# ── Stage 1: dependency builder ────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsndfile1-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime (CUDA 12.1 for GPU inference) ─────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

COPY --from=builder /install /usr/local
COPY --chown=appuser:appgroup . .

USER appuser

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD python3 -c "from app.health import check; check()"

# Default: run celery worker. Override per detector in docker-compose.
CMD ["celery", "-A", "app.worker", "worker", "--loglevel=info", "-Q", "text,image,audio,video,code", "--concurrency=2"]
