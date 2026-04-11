# Next Tasks

## Immediate (unblocks production)

1. Run `docker compose up -d --build backend worker`
2. Verify landing page loads at http://localhost:8000
3. Test text detection without login (paste text → analyze → result)
4. Test file uploads: image, audio, video
5. Check Celery queues receiving jobs (Flower at :5555)

## Short-term (post-launch)

- Deploy to Hetzner CX51 (8 vCPU, 32GB, ~30 EUR/month)
- Set up Caddy with real domain + auto-HTTPS
- Demo user job cleanup cron (delete jobs older than 24h)
- Monitor error rates via audit logs
- Push HF Space to huggingface.co

## Medium-term (model quality)

- Fine-tune DeBERTa on HC3/RAID datasets for text detection
- Train image classifier on AI-generated image datasets
- Train audio classifier on AI speech datasets
- Train video classifier on FakeAVCeleb/deepfake datasets
- Wire CodeDetector into code-specific worker (currently routes through text)
- Build accuracy benchmark dashboard

## Long-term (product growth)

- ONNX Runtime conversion for 2-4x inference speedup (ADR-006)
- GPU worker support for higher throughput
- Browser extension release
- API marketplace / developer portal
- Payment integration when product proves value
