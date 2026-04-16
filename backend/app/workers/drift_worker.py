"""
Daily drift detection — Celery Beat task that runs compute_daily_metrics
and compute_drift once per day, then logs PSI alerts.

Wraps the existing scripts/compute_daily_metrics.py and
scripts/compute_drift.py logic as importable functions. Does NOT
duplicate the math — imports directly from those modules.

Scheduled via celery_app.conf.beat_schedule (once per day at 02:00 UTC).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from .celery_app import celery_app

log = structlog.get_logger("drift_worker")

# PSI thresholds — must match scripts/compute_drift.py
PSI_WARN_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25


def _resolve_project_root() -> Path | None:
    """Find the project root that contains ai/text_detector/.

    Same discovery logic as text_worker.resolve_text_checkpoint_root().
    """
    here = os.path.dirname(__file__)
    candidates: list[str] = []
    env_root = os.environ.get("AUTHENTIC_PROJECT_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.append("/app")
    candidates.extend(
        os.path.abspath(os.path.join(here, *up))
        for up in ([], ["..", "..", ".."], ["..", "..", "..", ".."])
    )
    for root in candidates:
        if os.path.isdir(os.path.join(root, "ai", "text_detector")):
            return Path(root)
    return None


def _ensure_scripts_importable(root: Path) -> None:
    """Put the project root on sys.path so scripts/ are importable."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    scripts_str = str(root / "scripts")
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)


@celery_app.task(
    name="app.workers.drift_worker.run_daily_drift",
    queue="text",
    max_retries=1,
    default_retry_delay=300,
)
def run_daily_drift() -> dict[str, Any]:
    """Run daily metrics computation + PSI drift detection.

    Called by Celery Beat once per day. Writes:
      - metrics/daily_metrics.json (upserted for today's date)
      - metrics/drift.json (upserted for today's date)

    Returns a summary dict for Celery result inspection.
    """
    root = _resolve_project_root()
    if root is None:
        log.error("drift_worker_no_project_root")
        return {"error": "project root not found"}

    _ensure_scripts_importable(root)

    today = datetime.now(timezone.utc).date().isoformat()
    log_dir = root / "logs" / "predictions"
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"date": today}

    # ── Step 1: Daily metrics ───────────────────────────────
    try:
        from compute_daily_metrics import compute_metrics, _load_jsonl

        log_path = log_dir / f"{today}.jsonl"
        rows = _load_jsonl(log_path)
        metrics = compute_metrics(rows)
        metrics["date"] = today
        metrics["computed_at_utc"] = datetime.now(timezone.utc).isoformat()
        metrics["source_log"] = str(log_path).replace("\\", "/")

        output_path = metrics_dir / "daily_metrics.json"
        existing: dict[str, Any] = {}
        if output_path.exists():
            try:
                with output_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        if "days" not in existing:
            existing = {"days": {}, "schema_version": 1}
        existing["days"][today] = metrics
        existing["last_updated_utc"] = datetime.now(timezone.utc).isoformat()

        tmp = output_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp, output_path)

        result["daily_metrics"] = {
            "n_predictions": metrics.get("n_predictions", 0),
            "status": "ok",
        }
        log.info(
            "daily_metrics_computed",
            date=today,
            n_predictions=metrics.get("n_predictions", 0),
        )
    except Exception as exc:
        log.error("daily_metrics_failed", error=str(exc))
        result["daily_metrics"] = {"status": "error", "error": str(exc)}

    # ── Step 2: PSI drift detection ─────────────────────────
    try:
        from compute_drift import compute_psi, classify_psi, _bucketize, _load_jsonl as _load_drift_jsonl

        ref_path = root / "ai" / "text_detector" / "accuracy" / "training_distribution.json"
        if not ref_path.exists():
            log.warning("drift_reference_not_found", path=str(ref_path))
            result["drift"] = {"status": "skipped", "reason": "reference distribution not found"}
        else:
            with ref_path.open("r", encoding="utf-8") as f:
                ref_record = json.load(f)
            ref_edges = ref_record["psi"]["bucket_edges"]
            ref_counts = ref_record["psi"]["counts"]

            log_path = log_dir / f"{today}.jsonl"
            rows = _load_drift_jsonl(log_path)
            prod_probs = [
                float(r["meta_probability"])
                for r in rows
                if "meta_probability" in r and r["meta_probability"] is not None
            ]

            if not prod_probs:
                result["drift"] = {"status": "insufficient_data", "n_production": 0}
                log.info("drift_insufficient_data", date=today)
            else:
                prod_counts = _bucketize(prod_probs, ref_edges)
                psi, details = compute_psi(ref_counts, prod_counts)
                classification = classify_psi(psi)

                drift_record = {
                    "date": today,
                    "computed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "n_production": len(prod_probs),
                    "n_reference": int(sum(ref_counts)),
                    "psi": round(psi, 6),
                    "classification": classification,
                    "bucket_edges": ref_edges,
                    "per_bucket": details,
                    "thresholds": {
                        "stable_max": PSI_WARN_THRESHOLD,
                        "moderate_max": PSI_ALERT_THRESHOLD,
                    },
                    "reference_git_sha": ref_record.get("git_sha"),
                    "source_log": str(log_path).replace("\\", "/"),
                }

                # Write drift.json
                drift_output = metrics_dir / "drift.json"
                drift_existing: dict[str, Any] = {}
                if drift_output.exists():
                    try:
                        with drift_output.open("r", encoding="utf-8") as f:
                            drift_existing = json.load(f)
                    except Exception:
                        drift_existing = {}
                if "days" not in drift_existing:
                    drift_existing = {"days": {}, "schema_version": 1}
                drift_existing["days"][today] = drift_record
                drift_existing["last_updated_utc"] = datetime.now(timezone.utc).isoformat()

                tmp = drift_output.with_suffix(".json.tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(drift_existing, f, indent=2)
                os.replace(tmp, drift_output)

                # ── PSI alerting ────────────────────────────
                log.info(
                    "drift_psi_computed",
                    date=today,
                    psi=round(psi, 4),
                    classification=classification,
                    n_production=len(prod_probs),
                )

                if psi >= PSI_ALERT_THRESHOLD:
                    log.critical(
                        "ALERT_DRIFT_SIGNIFICANT",
                        psi=round(psi, 4),
                        classification=classification,
                        date=today,
                        action="Model retraining or investigation required",
                    )
                    _send_drift_alert(
                        severity="critical",
                        psi=psi,
                        classification=classification,
                        date=today,
                        n_production=len(prod_probs),
                    )
                elif psi >= PSI_WARN_THRESHOLD:
                    log.warning(
                        "ALERT_DRIFT_MODERATE",
                        psi=round(psi, 4),
                        classification=classification,
                        date=today,
                        action="Monitor closely — distribution shifting",
                    )
                    _send_drift_alert(
                        severity="warning",
                        psi=psi,
                        classification=classification,
                        date=today,
                        n_production=len(prod_probs),
                    )

                result["drift"] = {
                    "status": "ok",
                    "psi": round(psi, 4),
                    "classification": classification,
                    "n_production": len(prod_probs),
                }

    except Exception as exc:
        log.error("drift_computation_failed", error=str(exc))
        result["drift"] = {"status": "error", "error": str(exc)}

    return result


def _send_drift_alert(
    *,
    severity: str,
    psi: float,
    classification: str,
    date: str,
    n_production: int,
) -> None:
    """Post drift alert to the webhook (best-effort, never raises)."""
    try:
        from .alerting import _post_alert_webhook
        _post_alert_webhook(
            key=f"drift_{classification.lower()}",
            severity=severity,
            message=(
                f"Distribution drift detected: PSI={psi:.4f} ({classification}) "
                f"on {date} ({n_production} predictions). "
                f"Thresholds: warn>{PSI_WARN_THRESHOLD}, alert>{PSI_ALERT_THRESHOLD}."
            ),
            extra={
                "psi": round(psi, 4),
                "classification": classification,
                "date": date,
                "n_production": n_production,
            },
        )
    except Exception as exc:
        log.warning("drift_alert_webhook_failed", error=str(exc))
