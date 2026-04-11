# Pipeline Map — End-to-End Data Flow

## Text Detection
```
landing.html (runAnalysis('text'))
→ POST /api/v1/analyze/text { text, content_type: "text" }
→ routes.py:submit_text() → DetectionJob(status=PENDING)
→ run_text_detection.apply_async(queue="text")
→ text_worker.py → _get_detector()
  → tries: ai/text_detector/ensemble/text_detector.py (L1 perplexity + L2 stylometry)
  → fallback: _DevFallbackDetector (10-signal heuristic)
→ DetectionResult(authenticity_score, label, layer_scores, evidence, sentence_scores)
→ poll: GET /api/v1/jobs/{id} → status=completed
→ GET /api/v1/jobs/{id}/result → showResult(data)
```

## Image Detection
```
landing.html (drag-drop or file input)
→ POST /api/v1/analyze/file (multipart FormData)
→ routes.py:submit_file() → store_upload() → MinIO (ag-uploads)
→ celery_app.send_task("workers.image_worker.run_image_detection", queue="image")
→ image_worker.py → _get_detector() → ImageDetector
  → load_image → extract_all_features (GAN fingerprint, FFT, texture)
  → ImageClassifierEnsemble (EfficientNet-B4 + ViT-B/16, pretrained)
  → calibrate → ImageDetectionResult
→ DetectionResult(score, label, model_scores, evidence)
```

## Audio Detection
```
landing.html (file upload, audio tab)
→ POST /api/v1/analyze/file → MinIO → audio queue
→ audio_worker.py → AudioDetector
  → load_audio → chunk_audio (30s chunks)
  → per chunk: extract_features → AudioEnsemble (CNN + ResNet + Wav2Vec2)
  → aggregate: 60% max + 40% mean across chunks
→ DetectionResult + chunk_results (timeline with per-chunk scores)
```

## Video Detection
```
landing.html (file upload, video tab)
→ POST /api/v1/analyze/file → MinIO → video queue
→ video_worker.py → VideoDetector (timeout: 300s/360s)
  → ffmpeg extract frames at 2fps
  → FaceDetector (MediaPipe/MTCNN) per frame
  → per face: VideoClassifierEnsemble (XceptionNet + EfficientNet + ViT)
  → analyze_frame_artifacts + compute_temporal_features
  → aggregate: 50% classifier + 25% temporal + 15% artifacts + 10% max-frame
→ DetectionResult + frame_results (timeline with per-frame scores)
```

## Code Detection
```
landing.html (runAnalysis('code'))
→ POST /api/v1/analyze/text { text, content_type: "code" }
→ CONTENT_TYPE_TO_QUEUE maps code → "text" queue
→ text_worker.py (same as text, uses _DevFallbackDetector)
→ Future: ai/code_detector/code_detector.py (CodeBERT + AST, 60/40 split)
```

## URL Analysis
```
POST /api/v1/analyze/url { url }
→ url_analyzer.fetch_and_analyze_url() (SSRF-protected)
→ detect content type from response
→ text → store as input_text, route to text queue
→ binary → upload to MinIO, route to appropriate queue
```

## Queue Routing
- `CONTENT_TYPE_TO_QUEUE` in `celery_app.py`:
  - text → text, code → text, image → image, audio → audio, video → video
- Priority: `TIER_TO_PRIORITY`: free=1, pro=5, enterprise=9
- Timeouts: global 120s/180s soft/hard, video 300s/360s

## Frontend Polling
- `pollJob(jobId)`: max 150 attempts × 2s = 5 min timeout
- `fetchJobResult(jobId)`: maps backend response to UI format
- `showResult(data)`: renders score, verdict, layers, flags, sentences
- `authFetch()`: adds Bearer token if logged in, works without (demo mode)
