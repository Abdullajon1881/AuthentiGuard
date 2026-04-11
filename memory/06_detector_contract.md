# Detector Contract

Every AI detector and worker MUST follow these interfaces exactly.
Breaking this contract will cause silent failures or crashes.

## Detector Interface

Every detector class (`TextDetector`, `ImageDetector`, `AudioDetector`, `VideoDetector`, `CodeDetector`) MUST implement:

```python
class XxxDetector:
    def __init__(self, checkpoint_dir: Path | None = None, device: str | None = None):
        # checkpoint_dir is OPTIONAL — pretrained weights if None/missing
        # device is OPTIONAL — auto-detect CPU/GPU

    def load_models(self) -> None:
        # Called ONCE at worker startup
        # Loads pretrained weights first, then fine-tuned checkpoints IF dir exists
        # Must check: if self._checkpoint_dir and self._checkpoint_dir.exists()

    def analyze(self, data: bytes | str, filename: str) -> XxxDetectionResult:
        # Called per request
        # Returns dataclass with required fields (see below)
```

## Detection Result Fields (required)

| Field | Type | Description |
|-------|------|-------------|
| `score` | float [0,1] | AI probability (0=human, 1=AI) |
| `label` | str | "AI" / "HUMAN" / "UNCERTAIN" |
| `confidence` | float [0,1] | How sure the detector is |
| `model_scores` | dict[str, float] | Per-model/layer breakdown |
| `evidence` | dict[str, Any] | Supporting signals and features |
| `processing_ms` | int | Wall-clock time |

Additional per-modality:
- **Audio:** `chunk_results: list[ChunkResult]` (30s windows)
- **Video:** `frame_results: list[VideoFrameResult]` (per-frame scores)
- **Image:** `features: ImageFeatures` (extracted feature vector)

## Worker Contract

Workers extend `BaseDetectionWorker` in `backend/app/workers/base_worker.py`:

```python
class XxxDetectionWorker(BaseDetectionWorker):
    content_type = "xxx"  # must match ContentType enum

    def get_detector(self) -> Any:
        return _get_detector()  # singleton lazy-load

    async def get_input(self, job: DetectionJob) -> bytes | str:
        # Fetch from S3 or return job.input_text

    def run_detection(self, detector, input_data, job) -> Any:
        return detector.analyze(input_data, job.file_name)

    def build_result(self, job, detection_output, elapsed_ms) -> DetectionResult:
        # Map detector output to ORM DetectionResult
```

## Singleton Pattern (REQUIRED)

```python
_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        _detector = XxxDetector(checkpoint_dir=Path("ai/xxx_detector/checkpoints/phase3"))
        _detector.load_models()
    return _detector
```

## Error Handling
- `ValueError` = user's fault (bad input) → job FAILED, no retry
- Other exceptions → Celery retry (up to 3x with backoff)
- Optimistic locking via job version field prevents double-processing

## Dispatcher Normalization

`ai/ensemble_engine/routing/dispatcher.py`:
- `DetectorRegistry` maps content types to detector classes
- Lazy-loads detectors on first use
- All outputs normalized to `DetectorOutput(score, label, confidence, evidence)`
