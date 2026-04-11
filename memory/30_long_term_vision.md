# Long-Term Vision

## Product Roadmap

### v1 — Ship with pretrained fallbacks (current)
- All 5 modalities functional with pretrained weights + heuristic detectors
- Text detection works well (10-signal heuristic in _DevFallbackDetector)
- Image/audio/video return scores but accuracy is low (ImageNet/AudioSet weights, not AI detection)
- Hand-crafted forensic features (GAN fingerprint, FFT, spectral, temporal) provide real signal
- Code detection routes through text worker (placeholder)

### v2 — Fine-tuned classifiers
- DeBERTa fine-tuned on HC3/RAID datasets for text
- EfficientNet/ViT fine-tuned on AI-generated image datasets
- Audio CNN/ResNet fine-tuned on AI speech datasets
- XceptionNet fine-tuned on FakeAVCeleb for video
- XGBoost meta-classifier trained on ensemble outputs
- Accuracy benchmark dashboard with honest numbers

### v3 — Performance optimization
- ONNX Runtime for all detectors (ADR-006: 2-4x speedup)
- INT8 quantization where accuracy allows
- GPU worker pool for high-throughput deployments
- Batch processing API

### v4 — Product expansion
- Browser extension for real-time detection
- API marketplace with developer keys
- Webhook integrations for CMS/publishing platforms
- C2PA content credential verification

### v5 — Monetization
- Payment integration (free tier remains)
- Enterprise plans with SLA, priority queues, dedicated workers
- On-premise deployment option

## Distribution Strategy
- HuggingFace Space for ML community visibility (text-only Gradio demo)
- Design partner outreach for first 10 users (template in docs/outreach-template.md)
- Open source repo for trust and contributions
- Mission-driven: truth and authenticity first, revenue second

## Technical Debt to Address
- Single-file landing page (~3300 lines) → component extraction
- Alembic migrations (currently using create_all)
- Proper monitoring/alerting stack
- Load testing before scaling
