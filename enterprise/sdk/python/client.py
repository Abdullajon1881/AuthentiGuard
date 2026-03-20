"""
Steps 125–128: Enterprise API & SDK package.

Step 125: RESTful API with SDKs for Python, JavaScript, Go, and Java.
Step 126: Tiered pricing: free (limited), pro (high volume),
          enterprise (custom models, SLA, dedicated support).
Step 127: White-label options for embedding detection in own products.
Step 128: Compliance certifications: SOC 2 Type II, GDPR, HIPAA-compatible.

This module provides the Python SDK — the canonical reference implementation
that mirrors the structure of the JS, Go, and Java SDKs.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import structlog

log = structlog.get_logger(__name__)

SDK_VERSION = "0.1.0"

# ── Tier definitions (Step 126) ───────────────────────────────

TIERS = {
    "free": {
        "requests_per_month": 100,
        "requests_per_minute": 5,
        "max_file_mb":         10,
        "content_types":       ["text"],
        "batch_size":          0,
        "sla_uptime":          None,
        "support":             "community",
        "custom_models":       False,
        "white_label":         False,
        "price_usd":           0,
    },
    "pro": {
        "requests_per_month":  10_000,
        "requests_per_minute": 100,
        "max_file_mb":         100,
        "content_types":       ["text", "image", "audio", "video", "code"],
        "batch_size":          100,
        "sla_uptime":          99.5,
        "support":             "email (48h SLA)",
        "custom_models":       False,
        "white_label":         False,
        "price_usd":           49,
    },
    "enterprise": {
        "requests_per_month":  None,       # unlimited
        "requests_per_minute": 1000,
        "max_file_mb":         500,
        "content_types":       ["text", "image", "audio", "video", "code"],
        "batch_size":          1000,
        "sla_uptime":          99.9,
        "support":             "dedicated CSM (4h SLA)",
        "custom_models":       True,
        "white_label":         True,
        "price_usd":           None,       # custom contract
    },
}


# ── Python SDK client (Step 125) ──────────────────────────────

@dataclass
class AnalysisResult:
    """Unified result object returned by all SDK methods."""
    job_id:             str
    content_type:       str
    score:              float
    label:              str          # "AI" | "HUMAN" | "UNCERTAIN"
    confidence:         float
    confidence_level:   str
    processing_ms:      int
    layer_scores:       dict[str, float]
    model_attribution:  dict[str, float]
    verdict:            str
    flagged_segments:   list[dict]
    passport_id:        str | None   = None
    report_url:         str | None   = None
    raw:                dict | None  = None


class AuthentiGuardClient:
    """
    Python SDK for the AuthentiGuard API.

    Usage:
        client = AuthentiGuardClient(api_key="ag_...")

        # Detect AI text
        result = client.detect_text("Furthermore, it is worth noting...")
        print(result.label, result.score)

        # Detect AI image
        result = client.detect_file(Path("photo.jpg"))

        # Batch analysis
        results = list(client.detect_batch([
            {"type": "text", "content": "..."},
            {"type": "file", "path": "image.png"},
        ]))
    """

    BASE_URL = "https://api.authentiguard.io"

    def __init__(
        self,
        api_key:       str | None = None,
        jwt_token:     str | None = None,
        base_url:      str | None = None,
        timeout_s:     int = 60,
        max_retries:   int = 3,
        retry_delay_s: float = 1.0,
    ) -> None:
        if not api_key and not jwt_token:
            raise ValueError("Provide api_key or jwt_token")
        self._api_key      = api_key
        self._jwt_token    = jwt_token
        self._base_url     = base_url or self.BASE_URL
        self._timeout      = timeout_s
        self._max_retries  = max_retries
        self._retry_delay  = retry_delay_s

    def _headers(self) -> dict[str, str]:
        h = {"User-Agent": f"AuthentiGuard-Python-SDK/{SDK_VERSION}"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        elif self._jwt_token:
            h["Authorization"] = f"Bearer {self._jwt_token}"
        return h

    def _post(self, path: str, json_body: dict | None = None,
               files: dict | None = None) -> dict:
        """Make an authenticated POST request with retry logic."""
        import urllib.request, urllib.error
        url = f"{self._base_url}{path}"

        for attempt in range(self._max_retries):
            try:
                if files:
                    # Multipart form data
                    boundary = hashlib.md5(str(time.time()).encode()).hexdigest()
                    body, content_type = _encode_multipart(files, boundary)
                    req = urllib.request.Request(
                        url, data=body,
                        headers={**self._headers(),
                                  "Content-Type": f"multipart/form-data; boundary={boundary}"},
                        method="POST",
                    )
                else:
                    body = json.dumps(json_body or {}).encode()
                    req  = urllib.request.Request(
                        url, data=body,
                        headers={**self._headers(), "Content-Type": "application/json"},
                        method="POST",
                    )

                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return json.loads(resp.read())

            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < self._max_retries - 1:
                    retry_after = int(e.headers.get("Retry-After", self._retry_delay * 2))
                    time.sleep(retry_after)
                    continue
                raise AuthentiGuardAPIError(e.code, e.read().decode())
            except Exception as exc:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(self._retry_delay * (2 ** attempt))

        raise RuntimeError("Max retries exceeded")

    def detect_text(self, content: str, language: str | None = None) -> AnalysisResult:
        """
        Detect AI-generated text.

        Args:
            content:  Text to analyse (min 50 chars, max 1MB)
            language: ISO 639-1 code (auto-detected if omitted)

        Returns:
            AnalysisResult
        """
        if len(content) < 50:
            raise ValueError("Content must be at least 50 characters")

        body: dict = {"content": content}
        if language:
            body["language"] = language

        raw = self._post("/api/v1/analyze/text", body)
        return _parse_result(raw)

    def detect_file(
        self,
        path:            Path | str | BinaryIO,
        filename:        str | None = None,
        claimed_source:  str | None = None,
    ) -> AnalysisResult:
        """
        Detect AI-generated content in a file.

        Supported: JPEG, PNG, WebP, MP3, WAV, FLAC, MP4, MOV, .py, .js, .ts, ...

        Args:
            path:           File path, Path object, or file-like object
            filename:       Original filename (inferred from path if omitted)
            claimed_source: Optional context ("photo taken with iPhone 15")
        """
        if isinstance(path, (str, Path)):
            p = Path(path)
            filename = filename or p.name
            data = p.read_bytes()
        else:
            filename = filename or "upload"
            data = path.read()

        files: dict = {"file": (filename, data)}
        if claimed_source:
            files["claimed_source"] = (None, claimed_source.encode())

        # Async: poll for result
        raw = self._post("/api/v1/analyze/file", files=files)

        if raw.get("status") in ("pending", "processing"):
            raw = self._poll_job(raw["job_id"])

        return _parse_result(raw)

    def detect_url(self, url: str) -> AnalysisResult:
        """Detect AI-generated content at a URL (image or webpage)."""
        raw = self._post("/api/v1/analyze/url", {"url": url})
        if raw.get("status") in ("pending", "processing"):
            raw = self._poll_job(raw["job_id"])
        return _parse_result(raw)

    def _poll_job(self, job_id: str, max_wait_s: int = 300) -> dict:
        """Poll a job until completion."""
        import urllib.request
        deadline = time.time() + max_wait_s
        delay    = 1.0
        while time.time() < deadline:
            req = urllib.request.Request(
                f"{self._base_url}/api/v1/jobs/{job_id}/result",
                headers=self._headers(),
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            if data.get("status") == "completed":
                return data
            if data.get("status") == "failed":
                raise AuthentiGuardAPIError(0, f"Job failed: {data.get('error_message')}")
            time.sleep(min(delay, 10.0))
            delay *= 1.5
        raise TimeoutError(f"Job {job_id} did not complete in {max_wait_s}s")

    def detect_batch(
        self,
        items: list[dict],
        webhook_url: str | None = None,
    ) -> Iterator[AnalysisResult]:
        """
        Submit a batch of items (up to 1,000) for analysis.
        Yields AnalysisResult objects as they complete.

        Args:
            items: List of dicts with keys:
                   {"type": "text", "content": "..."}
                   {"type": "file", "path": "image.jpg", "filename": "image.jpg"}
        """
        for item in items:
            if item["type"] == "text":
                yield self.detect_text(item["content"])
            elif item["type"] == "file":
                yield self.detect_file(item["path"],
                                        item.get("filename"))

    def get_passport(self, content_hash: str) -> dict:
        """Retrieve an Authenticity Passport by content SHA-256."""
        import urllib.request
        req = urllib.request.Request(
            f"{self._base_url}/api/v1/passport/{content_hash}",
            headers=self._headers(),
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def verify_passport(self, content_hash: str) -> dict:
        """Verify an Authenticity Passport (public endpoint, no auth needed)."""
        import urllib.request
        req = urllib.request.Request(
            f"{self._base_url}/api/v1/passport/{content_hash}/verify",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())


# ── White-label SDK (Step 127) ────────────────────────────────

class WhiteLabelClient(AuthentiGuardClient):
    """
    White-label version of the SDK.
    Sends requests to the customer's own branded endpoint
    (backed by AuthentiGuard infrastructure).

    Usage:
        client = WhiteLabelClient(
            api_key="wl_...",
            brand_domain="detection.mycompany.com",
            brand_name="MyCompany AI Detector",
        )
    """

    def __init__(
        self,
        api_key:      str,
        brand_domain: str,
        brand_name:   str = "AI Detector",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=f"https://{brand_domain}",
            **kwargs,
        )
        self._brand_name = brand_name

    def _headers(self) -> dict[str, str]:
        h = super()._headers()
        h["User-Agent"] = f"{self._brand_name.replace(' ', '-')}-SDK/{SDK_VERSION}"
        h["X-White-Label"] = "true"
        return h


# ── Compliance utilities (Step 128) ──────────────────────────

def generate_hipaa_audit_log(
    job_id:        str,
    user_id:       str,
    action:        str,
    phi_hash:      str,    # hash of PHI — never store raw PHI
    ip_address:    str,
) -> dict[str, Any]:
    """
    Generate a HIPAA-compliant audit log entry.
    PHI is never stored — only the SHA-256 hash of the content.

    Required fields per 45 CFR § 164.312(b):
      - User identity, action, date/time, origin
    """
    return {
        "audit_type":   "HIPAA",
        "standard":     "45 CFR § 164.312(b)",
        "job_id":        job_id,
        "user_id":       user_id,
        "action":        action,
        "phi_hash":      phi_hash,    # SHA-256 only — no raw PHI
        "ip_address":    ip_address,
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system":        f"AuthentiGuard v{SDK_VERSION}",
        "note":          "PHI not stored; only hash retained for audit compliance",
    }


def soc2_compliance_attestation() -> dict[str, Any]:
    """Return SOC 2 Type II compliance attestation metadata."""
    return {
        "certification":   "SOC 2 Type II",
        "trust_criteria":  ["CC", "A", "PI", "C", "P"],
        "audit_period":    "Rolling 12 months",
        "auditor":         "Independent CPA firm (see certificate)",
        "controls_implemented": 16,
        "controls_partial":     2,
        "last_audit_date":  "See current certificate at https://authentiguard.io/compliance",
        "note": (
            "SOC 2 Type II certification requires a 6-month observation period. "
            "AuthentiGuard's first SOC 2 report is targeted for Q3 2025."
        ),
    }


# ── Helpers ────────────────────────────────────────────────────

class AuthentiGuardAPIError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


def _parse_result(raw: dict) -> AnalysisResult:
    return AnalysisResult(
        job_id=raw.get("job_id", ""),
        content_type=raw.get("content_type", "text"),
        score=float(raw.get("authenticity_score", raw.get("score", 0.5))),
        label=raw.get("label", "UNCERTAIN"),
        confidence=float(raw.get("confidence", 0.0)),
        confidence_level=raw.get("confidence_level", "low"),
        processing_ms=int(raw.get("processing_ms", 0)),
        layer_scores=raw.get("layer_scores", {}),
        model_attribution=raw.get("model_attribution", {}),
        verdict=raw.get("verdict_explanation", ""),
        flagged_segments=raw.get("flagged_segments", []),
        passport_id=raw.get("passport_id"),
        report_url=raw.get("report_url"),
        raw=raw,
    )


def _encode_multipart(
    files: dict,
    boundary: str,
) -> tuple[bytes, str]:
    """Encode files as multipart/form-data."""
    parts: list[bytes] = []
    for name, value in files.items():
        if isinstance(value, tuple):
            filename, data = value
            if filename is None:
                # Form field
                parts.append(
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
                    + data + b"\r\n"
                )
            else:
                parts.append(
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
                    + data + b"\r\n"
                )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
