# Project Brief

## Identity
- **Name:** AuthentiGuard
- **Type:** AI content authenticity detection platform
- **Stage:** Pre-product, solo developer startup
- **Mission:** Truth and authenticity in the era of AI-generated content
- **Business model:** Free/open source now, payment later when proven valuable

## What It Does
- Detects AI-generated content across 5 modalities: text, image, audio, video, code
- Returns evidence-based scores with per-layer breakdown
- Produces forensic reports with cryptographic signing

## Core Innovation
- Multi-layer ensemble per modality (not single-model)
- Each layer is independently trained/rule-based
- Attacker must defeat ALL layers simultaneously
- Calibrated probabilities (Platt + isotonic)
- Evidence transparency: every score has visible signals

## Supported Modalities
| Modality | Layers | Status |
|----------|--------|--------|
| Text | Perplexity (GPT-2) + Stylometry + DeBERTa + Adversarial | Heuristic fallback working |
| Image | EfficientNet-B4 + ViT-B/16 + GAN fingerprint + FFT + texture | Pretrained weights |
| Audio | Mel CNN + ResNet-18 + Wav2Vec2 + spectral features | Pretrained weights |
| Video | XceptionNet + EfficientNet + ViT + temporal + face forensics | Pretrained weights |
| Code | CodeBERT + AST pattern analysis | Pretrained weights |

## High-Level Pipeline
```
User input → API endpoint → Celery queue → Worker → Detector ensemble
→ Calibrate → Evidence collection → DetectionResult → Forensic report
```

## Scale
- 147 build steps, 22 phases, 260+ files, 360+ tests
- 20 API endpoints, 5 Celery queues, 8 Docker services
- Docker Compose (dev), K8s + Terraform (production AWS)
