"""
Step 85: C2PA provenance verification and ProvenanceRecord construction.

This module provides:
  - ProvenanceRecord dataclass (the canonical provenance output)
  - build_provenance_record() — called by the Authenticity Engine
  - C2PA manifest parsing from JPEG/PNG JUMBF boxes
  - Provenance chain reconstruction

C2PA embeds a cryptographically signed manifest into media files recording:
  who created the content, when, with what tools, and whether AI was involved.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ── Data classes ───────────────────────────────────────────────

@dataclass
class C2PAInfo:
    """Parsed C2PA manifest fields."""
    manifest_id:    str
    issuer:         str | None   # claim_generator
    creator:        str | None
    created_at:     str | None
    is_ai_generated: bool
    ai_model:       str | None
    content_hash:   str | None
    assertions:     list[dict]   # raw assertion list


@dataclass
class ProvenanceRecord:
    """Complete provenance record for one piece of content."""
    job_id:             str
    filename:           str
    content_hash:       str         # SHA-256 of raw bytes

    # C2PA
    has_c2pa:           bool
    c2pa_verified:      bool        # signature + hash verified
    c2pa:               C2PAInfo | None

    # Provenance chain
    provenance_chain:   list[dict]  # ordered list of steps

    # Tamper detection
    has_tamper_evidence: bool
    tamper_details:      list[str]

    # Metadata
    exif_present:       bool
    exif_summary:       dict[str, Any]


# ── Main builder ───────────────────────────────────────────────

def build_provenance_record(
    data:     bytes,
    filename: str,
    job_id:   str,
) -> ProvenanceRecord:
    """
    Build a complete ProvenanceRecord for the given raw content bytes.

    Steps:
      1. Hash the content
      2. Attempt C2PA parsing
      3. Verify C2PA signature
      4. Build provenance chain
      5. Check for tamper evidence
    """
    content_hash = hashlib.sha256(data).hexdigest()
    ext          = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    # C2PA parsing
    c2pa_info     = _try_parse_c2pa(data, filename, ext)
    c2pa_verified = False
    tamper_details: list[str] = []

    if c2pa_info:
        # Verify content hash if the manifest embeds one
        if c2pa_info.content_hash:
            if c2pa_info.content_hash.lower() != content_hash.lower():
                tamper_details.append(
                    "C2PA manifest content hash does not match current file hash — "
                    "file may have been modified after signing."
                )
            else:
                # Hash matches — manifest is internally consistent
                # Full signature verification requires c2pa-python + trust anchors
                c2pa_verified = _attempt_signature_verify(data, filename)
        else:
            # Manifest present but no hash — self-signed or incomplete
            c2pa_verified = False
            tamper_details.append("C2PA manifest lacks content hash — cannot fully verify.")

    # EXIF extraction (lightweight)
    exif_summary = _extract_exif_summary(data, ext)

    # Provenance chain
    chain = _build_chain(c2pa_info, content_hash, filename)

    return ProvenanceRecord(
        job_id=job_id,
        filename=filename,
        content_hash=content_hash,
        has_c2pa=c2pa_info is not None,
        c2pa_verified=c2pa_verified,
        c2pa=c2pa_info,
        provenance_chain=chain,
        has_tamper_evidence=len(tamper_details) > 0,
        tamper_details=tamper_details,
        exif_present=bool(exif_summary),
        exif_summary=exif_summary,
    )


# ── C2PA parsing ───────────────────────────────────────────────

def _try_parse_c2pa(data: bytes, filename: str, ext: str) -> C2PAInfo | None:
    """Attempt C2PA manifest parsing. Returns None if not present."""
    # Try official library first
    try:
        import c2pa  # type: ignore
        reader  = c2pa.Reader.from_bytes(filename, data)
        raw_json = json.loads(reader.json())
        return _c2pa_from_dict(raw_json)
    except ImportError:
        pass
    except Exception as exc:
        log.debug("c2pa_lib_parse_failed", error=str(exc)[:100])

    # Manual JUMBF scan
    try:
        if ext in ("jpg", "jpeg"):
            return _scan_jpeg_for_c2pa(data)
        if ext == "png":
            return _scan_png_for_c2pa(data)
    except Exception as exc:
        log.debug("manual_c2pa_scan_failed", error=str(exc)[:100])

    return None


def _scan_jpeg_for_c2pa(data: bytes) -> C2PAInfo | None:
    """Scan JPEG APP11 segments for JUMBF C2PA data."""
    i = 0
    while i < len(data) - 4:
        if data[i:i+2] == b"\xff\xeb":
            seg_len  = struct.unpack(">H", data[i+2:i+4])[0]
            seg_data = data[i+4:i+2+seg_len]
            if b"c2pa" in seg_data or b"jumb" in seg_data:
                return _extract_json_from_jumbf(seg_data)
        i += 1
    return None


def _scan_png_for_c2pa(data: bytes) -> C2PAInfo | None:
    """Scan PNG iTXt chunks for JUMBF C2PA data."""
    i = 8  # skip PNG signature
    while i < len(data) - 12:
        chunk_len  = struct.unpack(">I", data[i:i+4])[0]
        chunk_type = data[i+4:i+8]
        if chunk_type == b"iTXt":
            chunk_data = data[i+8:i+8+chunk_len]
            if b"C2PA" in chunk_data or b"c2pa" in chunk_data:
                return _extract_json_from_jumbf(chunk_data)
        i += 12 + chunk_len
    return None


def _extract_json_from_jumbf(data: bytes) -> C2PAInfo | None:
    """Extract JSON manifest from JUMBF box bytes."""
    j_start = data.find(b"{")
    j_end   = data.rfind(b"}")
    if j_start == -1 or j_end == -1:
        return None
    try:
        return _c2pa_from_dict(json.loads(data[j_start:j_end+1]))
    except Exception:
        return None


def _c2pa_from_dict(d: dict) -> C2PAInfo:
    """Build C2PAInfo from a parsed manifest dict."""
    claim      = d.get("claim", d)
    assertions_raw = claim.get("assertions", [])
    assertions = [{"label": a.get("label",""), "data": a.get("data",{})}
                  for a in assertions_raw]

    ai_assertion = next(
        (a["data"] for a in assertions
         if "ai" in a["label"].lower() or "generated" in a["label"].lower()),
        None,
    )

    return C2PAInfo(
        manifest_id=str(claim.get("instanceID", claim.get("manifest_id", "unknown"))),
        issuer=claim.get("claim_generator"),
        creator=claim.get("creator") or claim.get("dc:creator"),
        created_at=claim.get("created_at") or claim.get("dc:created"),
        is_ai_generated=bool(ai_assertion) or bool(claim.get("ai_generated")),
        ai_model=ai_assertion.get("model") if ai_assertion else None,
        content_hash=claim.get("hash") or claim.get("content_hash"),
        assertions=assertions,
    )


def _attempt_signature_verify(data: bytes, filename: str) -> bool:
    """Attempt full signature verification via c2pa-python."""
    try:
        import c2pa  # type: ignore
        reader = c2pa.Reader.from_bytes(filename, data)
        return reader.validation_status == "trusted"
    except Exception:
        return False


# ── Supporting helpers ─────────────────────────────────────────

def _extract_exif_summary(data: bytes, ext: str) -> dict[str, Any]:
    if ext not in ("jpg", "jpeg", "tiff", "tif"):
        return {}
    try:
        import exifread  # type: ignore
        import io
        tags = exifread.process_file(io.BytesIO(data), details=False)
        summary: dict[str, Any] = {}
        for key in ["Image Make", "Image Model", "EXIF DateTimeOriginal",
                     "Image Software", "GPS GPSLatitude"]:
            if key in tags:
                summary[key.lower().replace(" ", "_")] = str(tags[key])
        summary["has_exif"] = len(summary) > 0
        return summary
    except Exception:
        return {"has_exif": False}


def _build_chain(c2pa: C2PAInfo | None, content_hash: str, filename: str) -> list[dict]:
    chain: list[dict] = []
    if c2pa:
        if c2pa.creator:
            chain.append({
                "step":     "origin",
                "actor":    c2pa.creator,
                "tool":     c2pa.issuer or "unknown",
                "time":     c2pa.created_at,
                "verified": False,
            })
        for a in c2pa.assertions:
            if "edit" in a.get("label","").lower() or "action" in a.get("label","").lower():
                chain.append({
                    "step":   "edit",
                    "action": a["data"].get("action", "unknown"),
                    "tool":   a["data"].get("software_agent", "unknown"),
                })
        if c2pa.is_ai_generated:
            chain.append({
                "step":  "ai_generation",
                "model": c2pa.ai_model or "unspecified",
                "note":  "Content flagged as AI-generated in C2PA manifest",
            })
    chain.append({
        "step":         "authentiguard_analysis",
        "content_hash": content_hash,
        "filename":     filename,
    })
    return chain
