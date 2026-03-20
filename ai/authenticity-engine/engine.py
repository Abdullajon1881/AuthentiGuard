"""
Authenticity Engine — top-level orchestrator for Phase 11.

Wires together:
  Step 84: AuthenticityScore (unified multi-signal score)
  Step 85: C2PA provenance verification
  Step 86: ForensicReport generation
  Step 87: SHA-256 hashing + HMAC signing

Called by the backend Result Engine (backend/app/services/result_engine.py)
after all detectors have completed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from .scoring.authenticity_score import compute_authenticity_score, AuthenticityScore
from .provenance.c2pa import build_provenance_record, ProvenanceRecord
from .reports.report_generator import (
    generate_forensic_report, report_to_json, report_to_html, ForensicReport,
)
from .reports.integrity import hash_content, sign_report, create_integrity_block

log = structlog.get_logger(__name__)


@dataclass
class AuthenticityEngineResult:
    """Everything the backend needs to persist and return to the frontend."""
    # Score
    authenticity:    AuthenticityScore
    # Provenance
    provenance:      ProvenanceRecord
    # Report
    report:          ForensicReport
    report_json:     bytes
    report_html:     str
    # Integrity
    content_hash:    str
    report_hash:     str
    signature:       str
    integrity_block: dict[str, Any]


class AuthenticityEngine:
    """
    Orchestrates the full Phase 11 pipeline.
    Instantiated once per backend service; run() called per analysis job.
    """

    def __init__(self, secret_key: str) -> None:
        self._secret_key = secret_key

    def run(
        self,
        # Raw file
        raw_content:         bytes | str,
        filename:            str,
        content_type:        str,
        # Detection outputs
        ai_detection_score:  float,
        metadata_signals:    dict[str, Any],
        cross_modal_score:   float,
        # Report metadata
        job_id:              str,
        requested_by:        str,
        # Attribution and other signals
        model_attribution:   dict[str, float] | None = None,
        detector_outputs:    list[dict] | None = None,
        sentence_scores:     list[dict] | None = None,
        watermark:           dict[str, Any] | None = None,
        provenance_signals:  dict[str, Any] | None = None,
        processing_ms:       int = 0,
    ) -> AuthenticityEngineResult:
        """
        Full Phase 11 pipeline. Returns AuthenticityEngineResult.
        """
        raw_bytes = (
            raw_content if isinstance(raw_content, bytes)
            else raw_content.encode("utf-8")
        )
        prov_signals = provenance_signals or {}

        # ── Step 87a: Hash the content ────────────────────────
        content_hash = hash_content(raw_bytes)
        log.debug("content_hashed", hash_prefix=content_hash[:16])

        # ── Step 85: C2PA provenance verification ─────────────
        provenance = build_provenance_record(raw_bytes, filename, job_id)

        # Merge C2PA findings into provenance_signals for scoring
        if provenance.c2pa_verified:
            prov_signals["c2pa_verified"] = True
            if provenance.c2pa and provenance.c2pa.issuer:
                prov_signals["c2pa_issuer"] = provenance.c2pa.issuer
            if provenance.c2pa and provenance.c2pa.creator:
                prov_signals["c2pa_creator"] = provenance.c2pa.creator
        if provenance.has_tamper_evidence:
            prov_signals["tamper_detected"] = True

        # ── Step 84: Unified Authenticity Score ───────────────
        authenticity = compute_authenticity_score(
            ai_detection_score=ai_detection_score,
            metadata_signals=metadata_signals,
            provenance_signals=prov_signals,
            cross_modal_score=cross_modal_score,
            content_type=content_type,
        )
        log.info("authenticity_scored",
                 score=authenticity.score,
                 label=authenticity.label)

        # ── Step 86: Forensic report ──────────────────────────
        report = generate_forensic_report(
            job_id=job_id,
            content_type=content_type,
            filename=filename,
            authenticity=authenticity,
            ensemble_output={},
            provenance=provenance,
            watermark=watermark or {},
            model_attribution=model_attribution or {},
            detector_outputs=detector_outputs or [],
            sentence_scores=sentence_scores or [],
            requested_by=requested_by,
            processing_ms=processing_ms,
        )

        # ── Step 87b: Hash and sign the report ────────────────
        report_json  = report_to_json(report)
        report_hash  = hash_content(report_json)
        signature    = sign_report(report_json, self._secret_key)
        report_html  = report_to_html(report)

        # Attach integrity info to the report object
        report.report_hash      = report_hash
        report.report_signature = signature

        integrity_block = create_integrity_block(
            content_hash=content_hash,
            report_hash=report_hash,
            signature=signature,
            job_id=job_id,
            report_id=report.report_id,
        )

        log.info("authenticity_engine_complete",
                 job_id=job_id, score=authenticity.score,
                 report_id=report.report_id)

        return AuthenticityEngineResult(
            authenticity=authenticity,
            provenance=provenance,
            report=report,
            report_json=report_json,
            report_html=report_html,
            content_hash=content_hash,
            report_hash=report_hash,
            signature=signature,
            integrity_block=integrity_block,
        )
