"""
Unit tests for Phase 11 — Authenticity Engine.
Steps 84–87: unified scoring, C2PA parsing, report generation, integrity.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from ai.authenticity_engine.scoring.authenticity_score import (
    compute_authenticity_score, _ai_weight, _metadata_score,
    _provenance_score, _build_indicators, AuthenticityScore,
)
from ai.authenticity_engine.reports.integrity import (
    hash_content, hash_text, verify_content_hash,
    sign_report, verify_report_signature,
    create_integrity_block, hash_dict, is_duplicate_content,
)
from ai.authenticity_engine.reports.report_generator import (
    generate_forensic_report, report_to_json, report_to_html,
    _score_to_verdict, DISCLAIMER,
)
from ai.authenticity_engine.provenance.c2pa import (
    verify_c2pa, build_provenance_record,
)


# ── Step 84: Unified Authenticity Score ──────────────────────

class TestAuthenticityScore:
    def _meta(self, **kw) -> dict:
        return {"watermark": {}, "device_fingerprint": {}, "exif": {}, **kw}

    def _prov(self, **kw) -> dict:
        return {**kw}

    def test_high_ai_score_labels_ai(self) -> None:
        result = compute_authenticity_score(0.90, self._meta(), self._prov())
        assert result.label == "AI"
        assert result.score >= 0.75

    def test_low_ai_score_labels_human(self) -> None:
        result = compute_authenticity_score(0.10, self._meta(), self._prov())
        assert result.label == "HUMAN"
        assert result.score <= 0.40

    def test_score_always_in_range(self) -> None:
        for ai in [0.0, 0.01, 0.5, 0.99, 1.0]:
            r = compute_authenticity_score(ai, self._meta(), self._prov())
            assert 0.0 < r.score < 1.0, f"Score {r.score} out of range for ai={ai}"

    def test_watermark_raises_score(self) -> None:
        base = compute_authenticity_score(0.70, self._meta(), self._prov())
        with_wm = compute_authenticity_score(
            0.70,
            self._meta(watermark={"watermark_detected": True, "confidence": 0.9}),
            self._prov(),
        )
        assert with_wm.score > base.score

    def test_c2pa_verified_lowers_score(self) -> None:
        base  = compute_authenticity_score(0.65, self._meta(), self._prov())
        c2pa_ = compute_authenticity_score(
            0.65, self._meta(), self._prov(c2pa_verified=True)
        )
        assert c2pa_.score < base.score

    def test_c2pa_ai_creator_raises_score(self) -> None:
        base  = compute_authenticity_score(0.65, self._meta(), self._prov())
        c2pa_ = compute_authenticity_score(
            0.65, self._meta(),
            self._prov(c2pa_verified=True, c2pa_creator="openai/dalle-3"),
        )
        assert c2pa_.score > base.score

    def test_cross_modal_adjustment_applied(self) -> None:
        no_cm = compute_authenticity_score(0.60, self._meta(), self._prov(), 0.0)
        with_cm = compute_authenticity_score(0.60, self._meta(), self._prov(), 1.0)
        assert with_cm.score > no_cm.score

    def test_signal_contributions_present(self) -> None:
        result = compute_authenticity_score(0.80, self._meta(), self._prov())
        assert len(result.signal_contributions) >= 3
        names = [c.name for c in result.signal_contributions]
        assert "ai_detection" in names

    def test_confidence_formula(self) -> None:
        for score in [0.01, 0.40, 0.50, 0.75, 0.99]:
            r = compute_authenticity_score(score, self._meta(), self._prov())
            assert 0.0 <= r.confidence <= 1.0
        r_mid = compute_authenticity_score(0.50, self._meta(), self._prov())
        assert r_mid.confidence < 0.2   # near 0.5 → low confidence

    def test_ai_weight_by_content_type(self) -> None:
        assert _ai_weight("text") > _ai_weight("video")
        assert _ai_weight("code") == _ai_weight("text")

    def test_metadata_score_no_signals(self) -> None:
        score, weight = _metadata_score({}, "text")
        assert weight == 0.0   # no metadata → don't contribute

    def test_metadata_score_ai_software(self) -> None:
        exif = {"image_software": "Stable Diffusion v2.1", "has_exif": True}
        score, weight = _metadata_score({"exif": exif}, "image")
        assert score > 0.80
        assert weight > 0.0

    def test_metadata_score_camera_exif(self) -> None:
        dev = {"likely_camera_capture": True, "likely_ai_generated": False}
        score, weight = _metadata_score({"device_fingerprint": dev}, "image")
        assert score < 0.20   # camera EXIF = human signal

    def test_provenance_score_c2pa_human(self) -> None:
        score, weight = _provenance_score({"c2pa_verified": True})
        assert score < 0.15
        assert weight > 0.0

    def test_provenance_score_empty(self) -> None:
        score, weight = _provenance_score({})
        assert weight == 0.0

    def test_trust_indicators_populated(self) -> None:
        trust, _ = _build_indicators(
            {"device_fingerprint": {"likely_camera_capture": True}, "exif": {}, "watermark": {}},
            {"c2pa_verified": True},
            ai_score=0.20, final_score=0.15,
        )
        assert any("C2PA" in t["indicator"] for t in trust)

    def test_risk_indicators_populated(self) -> None:
        _, risk = _build_indicators(
            {"device_fingerprint": {"likely_ai_generated": True, "suspicious_signals": ["no_exif"]},
             "exif": {}, "watermark": {"watermark_detected": True, "z_score": 4.2, "confidence": 0.8}},
            {},
            ai_score=0.90, final_score=0.90,
        )
        assert any("watermark" in r["indicator"].lower() for r in risk)


# ── Step 87: Integrity (hashing + signing) ───────────────────

class TestIntegrity:
    def test_hash_content_sha256(self) -> None:
        data   = b"hello world"
        result = hash_content(data)
        expected = hashlib.sha256(data).hexdigest()
        assert result == expected

    def test_hash_content_length(self) -> None:
        assert len(hash_content(b"test")) == 64   # hex SHA-256

    def test_hash_text_utf8(self) -> None:
        assert hash_text("hello") == hash_content("hello".encode())

    def test_verify_content_hash_valid(self) -> None:
        data = b"test content"
        h    = hash_content(data)
        assert verify_content_hash(data, h) is True

    def test_verify_content_hash_invalid(self) -> None:
        data = b"test content"
        assert verify_content_hash(data, "0" * 64) is False

    def test_verify_content_hash_case_insensitive(self) -> None:
        data = b"test"
        h    = hash_content(data).upper()
        assert verify_content_hash(data, h) is True

    def test_sign_report_produces_hex(self) -> None:
        sig = sign_report(b"test content", "secret-key")
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_sign_report_deterministic(self) -> None:
        sig1 = sign_report(b"content", "key")
        sig2 = sign_report(b"content", "key")
        assert sig1 == sig2

    def test_sign_report_different_keys_differ(self) -> None:
        s1 = sign_report(b"content", "key1")
        s2 = sign_report(b"content", "key2")
        assert s1 != s2

    def test_verify_report_signature_valid(self) -> None:
        content = b"report data"
        key     = "platform-secret"
        sig     = sign_report(content, key)
        assert verify_report_signature(content, sig, key) is True

    def test_verify_report_signature_tampered_content(self) -> None:
        sig = sign_report(b"original content", "key")
        assert verify_report_signature(b"tampered content", sig, "key") is False

    def test_verify_report_signature_wrong_key(self) -> None:
        sig = sign_report(b"content", "correct-key")
        assert verify_report_signature(b"content", sig, "wrong-key") is False

    def test_integrity_block_has_required_fields(self) -> None:
        block = create_integrity_block("abc", "def", "sig", "job1", "rep1")
        required = {"content_hash", "report_hash", "signature",
                     "job_id", "report_id", "signed_at", "algorithm"}
        assert required.issubset(block.keys())

    def test_hash_dict_deterministic(self) -> None:
        d = {"b": 2, "a": 1}
        assert hash_dict(d) == hash_dict(d)
        assert hash_dict({"a": 1, "b": 2}) == hash_dict({"b": 2, "a": 1})

    def test_is_duplicate_content(self) -> None:
        h = hash_content(b"data")
        assert is_duplicate_content(h, [h, "other"]) is True
        assert is_duplicate_content(h, ["different", "hashes"]) is False

    def test_is_duplicate_case_insensitive(self) -> None:
        h = hash_content(b"data")
        assert is_duplicate_content(h.upper(), [h]) is True


# ── Step 86: Report generation ───────────────────────────────

class TestReportGeneration:
    def _make_authenticity(self) -> AuthenticityScore:
        return compute_authenticity_score(
            0.85,
            {"watermark": {}, "device_fingerprint": {}, "exif": {}},
            {},
        )

    def _make_provenance(self) -> Any:
        """Minimal provenance-like object."""
        from ai.authenticity_engine.provenance.c2pa import ProvenanceRecord, C2PAManifest
        c2pa = C2PAManifest(
            is_valid=False, is_present=False, creator=None, tool=None,
            timestamp=None, content_hash=None, hash_matches=None,
            issuer=None, actions=[], assertions=[], error=None,
        )
        return ProvenanceRecord(
            content_hash="a" * 64, c2pa=c2pa, c2pa_verified=False,
        )

    def test_generate_report_returns_forensic_report(self) -> None:
        from ai.authenticity_engine.reports.report_generator import ForensicReport
        auth = self._make_authenticity()
        prov = self._make_provenance()
        report = generate_forensic_report(
            job_id="job-123", content_type="text", filename="test.txt",
            authenticity=auth, ensemble_output={}, provenance=prov,
            watermark={}, model_attribution={}, detector_outputs=[],
            sentence_scores=[], requested_by="test@example.com", processing_ms=500,
        )
        assert isinstance(report, ForensicReport)
        assert report.job_id == "job-123"
        assert report.content_type == "text"

    def test_report_verdict(self) -> None:
        assert _score_to_verdict(0.80) == "AI-GENERATED"
        assert _score_to_verdict(0.30) == "AUTHENTIC"
        assert _score_to_verdict(0.55) == "UNCERTAIN"

    def test_report_to_json_is_valid_json(self) -> None:
        auth = self._make_authenticity()
        prov = self._make_provenance()
        report = generate_forensic_report(
            job_id="j1", content_type="text", filename=None,
            authenticity=auth, ensemble_output={}, provenance=prov,
            watermark={}, model_attribution={}, detector_outputs=[],
            sentence_scores=[], requested_by="user", processing_ms=100,
        )
        data = report_to_json(report)
        parsed = json.loads(data)
        assert parsed["job_id"] == "j1"

    def test_report_json_contains_disclaimer(self) -> None:
        auth = self._make_authenticity()
        prov = self._make_provenance()
        report = generate_forensic_report(
            job_id="j2", content_type="text", filename=None,
            authenticity=auth, ensemble_output={}, provenance=prov,
            watermark={}, model_attribution={}, detector_outputs=[],
            sentence_scores=[], requested_by="user", processing_ms=100,
        )
        data = report_to_json(report)
        assert b"disclaimer" in data.lower()

    def test_report_to_html_contains_verdict(self) -> None:
        auth = self._make_authenticity()
        prov = self._make_provenance()
        report = generate_forensic_report(
            job_id="j3", content_type="text", filename=None,
            authenticity=auth, ensemble_output={}, provenance=prov,
            watermark={}, model_attribution={}, detector_outputs=[],
            sentence_scores=[], requested_by="user", processing_ms=100,
        )
        html = report_to_html(report)
        assert "<!DOCTYPE html>" in html
        assert "AI-GENERATED" in html

    def test_sign_and_verify_report(self) -> None:
        auth = self._make_authenticity()
        prov = self._make_provenance()
        report = generate_forensic_report(
            job_id="j4", content_type="text", filename=None,
            authenticity=auth, ensemble_output={}, provenance=prov,
            watermark={}, model_attribution={}, detector_outputs=[],
            sentence_scores=[], requested_by="user", processing_ms=100,
        )
        report_bytes = report_to_json(report)
        sig  = sign_report(report_bytes, "test-secret")
        assert verify_report_signature(report_bytes, sig, "test-secret") is True
        assert verify_report_signature(report_bytes, sig, "wrong-secret") is False

    def test_disclaimer_is_not_empty(self) -> None:
        assert len(DISCLAIMER) > 50


# ── Step 85: C2PA ─────────────────────────────────────────────

class TestC2PA:
    def test_verify_c2pa_unknown_file_returns_not_present(self) -> None:
        result = verify_c2pa(b"not a real image file", "test.jpg")
        assert result.is_present is False

    def test_build_provenance_record_computes_hash(self) -> None:
        data   = b"test image data"
        record = build_provenance_record(data, "test.jpg", "job-123")
        expected_hash = hashlib.sha256(data).hexdigest()
        assert record.content_hash == expected_hash

    def test_build_provenance_record_file_size(self) -> None:
        data   = b"x" * 1000
        record = build_provenance_record(data, "test.png")
        assert record.file_size == 1000

    def test_build_provenance_no_c2pa_strength_none(self) -> None:
        data   = b"plain file data"
        record = build_provenance_record(data, "test.txt")
        assert record.provenance_strength == "none"
        assert record.c2pa_verified is False

    def test_no_tamper_for_clean_file(self) -> None:
        data   = b"clean image"
        record = build_provenance_record(data, "clean.jpg")
        assert record.has_tamper_evidence is False
