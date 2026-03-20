"""
Step 55: Video upload and frame extraction pipeline.

Extracts frames at 1 frame per 0.5s (2 fps) using ffmpeg.
This rate captures temporal inconsistencies without processing
every frame (which would be 24–60× slower).

Pipeline:
  raw bytes → validate → ffmpeg decode → frame extraction →
  [BGR frames at 2fps] → face detection per frame
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────
FRAMES_PER_SECOND   = 2.0          # Step 55: 1 frame per 0.5s
MAX_VIDEO_DURATION  = 600.0        # 10 minutes max
MIN_VIDEO_DURATION  = 1.0          # reject clips under 1s
MAX_FRAMES          = 1200         # safety cap (10min × 2fps)
TARGET_FRAME_SIZE   = (224, 224)   # resize for CNN input

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}


@dataclass
class VideoFrame:
    """One decoded video frame with its metadata."""
    frame_idx:   int
    timestamp_s: float
    image:       np.ndarray    # shape [H, W, 3] BGR uint8
    width:       int
    height:      int


@dataclass
class VideoMetadata:
    """Basic video file metadata extracted by ffprobe."""
    duration_s:  float
    fps:         float
    width:       int
    height:      int
    codec:       str
    n_frames_extracted: int
    filename:    str


# ── ffprobe metadata ──────────────────────────────────────────

def probe_video(data: bytes, filename: str) -> VideoMetadata:
    """
    Use ffprobe to extract video metadata without decoding frames.
    Falls back to safe defaults if ffprobe is unavailable.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {ext}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                tmp_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return _fallback_metadata(filename)

        import json
        info = json.loads(result.stdout)

        # Find the video stream
        video_stream = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if not video_stream:
            return _fallback_metadata(filename)

        duration = float(info.get("format", {}).get("duration", 0.0))

        # Parse fps from avg_frame_rate (e.g., "24/1" or "30000/1001")
        fps_str = video_stream.get("avg_frame_rate", "0/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / max(float(den), 1)
        except (ValueError, ZeroDivisionError):
            fps = 25.0

        return VideoMetadata(
            duration_s=duration,
            fps=fps,
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            codec=video_stream.get("codec_name", "unknown"),
            n_frames_extracted=0,   # set after extraction
            filename=filename,
        )
    except Exception as exc:
        log.warning("ffprobe_failed", error=str(exc))
        return _fallback_metadata(filename)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _fallback_metadata(filename: str) -> VideoMetadata:
    return VideoMetadata(
        duration_s=0.0, fps=25.0, width=0, height=0,
        codec="unknown", n_frames_extracted=0, filename=filename,
    )


# ── Frame extraction ──────────────────────────────────────────

def extract_frames(
    data: bytes,
    filename: str,
    fps: float = FRAMES_PER_SECOND,
) -> tuple[list[VideoFrame], VideoMetadata]:
    """
    Step 55: Extract frames at `fps` using ffmpeg.
    Returns (frames, metadata). Frames are BGR uint8 numpy arrays.

    Uses pipe-based extraction to avoid disk I/O overhead.
    Falls back to OpenCV VideoCapture if ffmpeg is unavailable.
    """
    ext = Path(filename).suffix.lower()

    # Try ffmpeg pipe first
    try:
        return _extract_with_ffmpeg(data, ext, fps, filename)
    except Exception as e:
        log.warning("ffmpeg_extraction_failed", error=str(e))

    # Fallback: OpenCV
    try:
        return _extract_with_opencv(data, ext, fps, filename)
    except Exception as e:
        raise RuntimeError(f"All frame extractors failed for {filename}: {e}") from e


def _extract_with_ffmpeg(
    data: bytes, ext: str, fps: float, filename: str,
) -> tuple[list[VideoFrame], VideoMetadata]:
    """Extract frames using ffmpeg pipe — no temp files."""
    import struct

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        meta = probe_video(data, filename)

        # Validate duration
        if meta.duration_s < MIN_VIDEO_DURATION:
            raise ValueError(f"Video too short: {meta.duration_s:.1f}s")

        duration = min(meta.duration_s, MAX_VIDEO_DURATION)
        w, h = TARGET_FRAME_SIZE

        cmd = [
            "ffmpeg", "-i", tmp_path,
            "-vf", f"fps={fps},scale={w}:{h}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-t", str(duration),
            "-",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, timeout=120,
        )

        if proc.returncode != 0 and not proc.stdout:
            raise RuntimeError(f"ffmpeg error: {proc.stderr.decode()[:200]}")

        # Parse raw video frames from stdout
        frame_size = w * h * 3
        raw = proc.stdout
        frames: list[VideoFrame] = []

        for i in range(min(len(raw) // frame_size, MAX_FRAMES)):
            offset = i * frame_size
            arr = np.frombuffer(raw[offset:offset + frame_size], dtype=np.uint8)
            if len(arr) < frame_size:
                break
            image = arr.reshape(h, w, 3)
            frames.append(VideoFrame(
                frame_idx=i,
                timestamp_s=round(i / fps, 3),
                image=image,
                width=w,
                height=h,
            ))

        meta.n_frames_extracted = len(frames)
        log.info("frames_extracted_ffmpeg",
                 filename=filename, n=len(frames), fps=fps)
        return frames, meta

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _extract_with_opencv(
    data: bytes, ext: str, fps: float, filename: str,
) -> tuple[list[VideoFrame], VideoMetadata]:
    """Fallback frame extraction using OpenCV."""
    import cv2  # type: ignore

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open video")

        src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration  = src_count / src_fps
        step      = int(src_fps / fps)   # sample every N-th frame
        w, h      = TARGET_FRAME_SIZE

        frames: list[VideoFrame] = []
        frame_idx = 0
        extracted = 0

        while extracted < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                resized = cv2.resize(frame, (w, h))
                frames.append(VideoFrame(
                    frame_idx=extracted,
                    timestamp_s=round(frame_idx / src_fps, 3),
                    image=resized,
                    width=w,
                    height=h,
                ))
                extracted += 1
            frame_idx += 1

        cap.release()
        meta = VideoMetadata(
            duration_s=duration, fps=src_fps, width=w, height=h,
            codec="opencv", n_frames_extracted=len(frames), filename=filename,
        )
        log.info("frames_extracted_opencv", filename=filename, n=len(frames))
        return frames, meta

    finally:
        Path(tmp_path).unlink(missing_ok=True)
