# Known Issues

## Critical (blocks production)

- **No fine-tuned checkpoints for any detector**
  - All 5 detectors fall back to pretrained weights or heuristics
  - Text: works well via _DevFallbackDetector (10-signal heuristic)
  - Image/audio/video: pretrained ImageNet/AudioSet weights don't detect AI content
  - Hand-crafted features (GAN fingerprint, FFT, spectral) provide real forensic signal
  - Code: CodeBERT pretrained + AST heuristics provide some signal

- **Docker worker build fragile**
  - `libgl1-mesa-glx` removed in Debian Trixie → use `libgl1` (FIXED in code)
  - `spacy==3.7.4` conflicts with `fastapi-cli` typer dep → loosened to `>=3.7.4,<4.0` (FIXED)
  - Build takes 10-15 min (PyTorch ~800MB + model downloads)
  - Docker daemon can drop connection during large layer export

## Important (degrades experience)

- **Demo user job accumulation**
  - Anonymous demo requests create jobs owned by demo user
  - No cleanup cron yet → unbounded growth
  - Need: periodic cleanup of demo user jobs older than 24h

- **Pretrained classifier accuracy**
  - EfficientNet/ViT/XceptionNet trained on ImageNet, not AI detection
  - Classifier scores are near-random for AI detection task
  - Hand-crafted features (FFT, GAN fingerprint, texture, temporal) carry the actual signal
  - Accuracy will be poor until models are fine-tuned on detection datasets

- **CSP header restrictive in production**
  - Must allow Google Fonts (fonts.googleapis.com, fonts.gstatic.com)
  - Must allow 'unsafe-inline' for landing page styles
  - May need updates if landing page adds external resources

## Minor

- **Code detection routes through text worker**
  - `CONTENT_TYPE_TO_QUEUE` maps code → text queue
  - Uses same _DevFallbackDetector as text (no code-specific analysis yet)
  - CodeDetector exists at ai/code_detector but not wired into worker

- **Single-file landing page**
  - ~3300 lines of HTML/CSS/JS in one file
  - Works but hard to maintain
  - No build step, no bundler, no component extraction

- **ONNX not yet used**
  - ADR-006 specifies ONNX Runtime for production inference
  - All models currently run in PyTorch mode
  - ONNX conversion deferred until models are fine-tuned

## Risks That Cause AI Agent Failures
- Rewriting architecture without reading memory files
- Introducing new frameworks alongside existing ones
- Duplicating worker/detector components
- Removing required pipeline steps (ensemble → calibrate → evidence)
- Breaking async processing by making endpoints synchronous
- Pinning exact versions of spacy (causes typer conflicts)
