# Current Focus

**Phase:** Detection accuracy improvement
**Date:** 2026-04-12

## Status
- Docker optimization COMPLETE (3-stage build, CPU-only PyTorch, ~1.5GB image)
- Accuracy roadmap DELIVERED (4-phase plan)
- Current detection: text ~0.60 F1 (heuristic fallback), image/audio/video ~0.50 (untrained)

## Accuracy Roadmap Summary
1. **Phase 1 (1-2 days):** Activate real TextDetector (L1+L2), tune thresholds → text F1 ~0.75-0.80
2. **Phase 2 (3-5 days):** Fine-tune DeBERTa-v3-base on HC3/RAID/OpenGPTText → text F1 ~0.85-0.90
3. **Phase 3 (2-3 weeks):** Fine-tune image (EfficientNet-B4 on CIFAKE/GenImage) + audio (Wav2Vec2 on ASVspoof) → image F1 0.80, audio F1 0.85
4. **Phase 4 (1 week):** Benchmark suite, Platt calibration, CI integration

## Next Action
1. Phase 1A: Fix `_get_detector()` in `text_worker.py` to use real TextDetector instead of _DevFallbackDetector
2. Phase 1B: Run 200 samples through L1 perplexity, tune thresholds
3. Phase 2A: Download HC3 + RAID datasets, prepare parquet files
