"""
Stage 3 — Part 1: prediction logging for drift + audit.

`log_prediction()` is called from the Celery worker AFTER
`TextDetector.analyze()` returns. It writes two files per day:

  logs/predictions/YYYY-MM-DD.jsonl        — every prediction, no text
  logs/predictions/YYYY-MM-DD.samples.jsonl — probabilistic sample, with text

The predictions log is the input to `scripts/compute_daily_metrics.py`
and `scripts/compute_drift.py`. The samples log is the input to
`scripts/sample_predictions.py` (or direct manual audit).

Both writes are append-only JSONL. Small single-line writes are atomic
on POSIX (< PIPE_BUF) and on Windows for sub-kilobyte payloads, which
our records always are. Multiple worker processes can share the files
safely. No explicit locking.

Every public call is exception-safe: a logging failure emits a
structlog warning and returns. The caller's prediction return value
is never affected.

Config knobs:
  PREDICTION_LOG_DIR       override the default `logs/predictions` path
  PREDICTION_SAMPLE_RATE   float in [0.0, 1.0]; default 0.01 (~1%)
  PREDICTION_LOG_ENABLED   "1" (default) or "0" to disable entirely
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ── Defaults ─────────────────────────────────────────────────────────
# Repo-root default — workers start in /app on Docker but the tests
# and scripts run from the repo root. Callers can override via the
# `PREDICTION_LOG_DIR` env var.
_DEFAULT_LOG_DIR = Path("logs") / "predictions"
_DEFAULT_SAMPLE_RATE = 0.01

_TEXT_TRUNCATE_CHARS = 2000  # cap stored text to bound log size


def _log_dir() -> Path:
    override = os.environ.get("PREDICTION_LOG_DIR")
    if override:
        return Path(override)
    return _DEFAULT_LOG_DIR


def _sample_rate() -> float:
    try:
        raw = os.environ.get("PREDICTION_SAMPLE_RATE")
        if raw is None:
            return _DEFAULT_SAMPLE_RATE
        r = float(raw)
        if r < 0.0:
            return 0.0
        if r > 1.0:
            return 1.0
        return r
    except Exception:
        return _DEFAULT_SAMPLE_RATE


def _enabled() -> bool:
    return os.environ.get("PREDICTION_LOG_ENABLED", "1") != "0"


def _layer_score(layer_results: Any, name: str) -> float | None:
    """Extract a layer score by name. Returns None if missing or errored."""
    for r in layer_results:
        if getattr(r, "layer_name", None) == name:
            if getattr(r, "error", None):
                return None
            return float(getattr(r, "score", 0.5))
    return None


def _build_record(
    *,
    text: str,
    result: Any,
    model_version: str,
    include_text: bool,
) -> dict[str, Any]:
    """Construct the record dict — shared shape between the main log
    and the samples log, except the samples log carries the text.

    Every field is explicitly typed and None-safe so a broken
    LayerResult (one of them missing, etc.) cannot poison the log.
    """
    layer_results = getattr(result, "layer_results", []) or []
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "input_length": int(len(text) if isinstance(text, str) else 0),
        "l1_score": _layer_score(layer_results, "perplexity"),
        "l2_score": _layer_score(layer_results, "stylometry"),
        "l3_score": _layer_score(layer_results, "transformer"),
        "meta_probability": float(getattr(result, "score", 0.5)),
        "final_label": str(getattr(result, "label", "UNCERTAIN")),
    }
    if include_text and isinstance(text, str):
        # Cap stored text to bound sample-log size. The tail is kept
        # as well in case the prompt-injection marker lives there.
        if len(text) <= _TEXT_TRUNCATE_CHARS:
            record["text"] = text
        else:
            record["text"] = (
                text[: _TEXT_TRUNCATE_CHARS // 2]
                + "\n...[TRUNCATED]...\n"
                + text[-_TEXT_TRUNCATE_CHARS // 2 :]
            )
            record["text_truncated"] = True
    return record


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Atomic-ish append of a single JSONL record.

    Records are < 1 KB which is well under PIPE_BUF on POSIX and well
    under the atomic-write threshold on Windows NTFS. Multi-process
    concurrent appends are safe without an explicit lock for records
    of this size.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def log_prediction(
    *,
    text: str,
    result: Any,
    model_version: str,
    now: datetime | None = None,
    rand: random.Random | None = None,
) -> None:
    """Log a prediction to the daily JSONL files.

    Never raises. Any failure emits a structlog warning and returns
    without affecting the caller's return value.

    Arguments are keyword-only so future additions don't break callers.
    `now` and `rand` are dependency-injected for deterministic tests;
    production code passes neither.
    """
    if not _enabled():
        return

    try:
        now_dt = now or datetime.now(timezone.utc)
        today_iso = now_dt.date().isoformat()
        log_dir = _log_dir()

        # Main log — every prediction, no text
        main_record = _build_record(
            text=text,
            result=result,
            model_version=model_version,
            include_text=False,
        )
        # Override the record timestamp if `now` was injected (tests)
        if now is not None:
            main_record["timestamp"] = now_dt.isoformat()
        main_path = log_dir / f"{today_iso}.jsonl"
        _append_jsonl(main_path, main_record)

        # Samples log — probabilistic, with text
        rng = rand or random
        rate = _sample_rate()
        if rate > 0.0 and rng.random() < rate:
            sample_record = _build_record(
                text=text,
                result=result,
                model_version=model_version,
                include_text=True,
            )
            if now is not None:
                sample_record["timestamp"] = now_dt.isoformat()
            sample_record["sample_rate"] = rate
            sample_path = log_dir / f"{today_iso}.samples.jsonl"
            _append_jsonl(sample_path, sample_record)
    except Exception as exc:
        # Logging must NEVER break the prediction response
        log.warning("prediction_log_failed", error=str(exc))
