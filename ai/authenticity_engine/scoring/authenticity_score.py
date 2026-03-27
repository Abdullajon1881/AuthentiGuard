"""
Step 84: Unified Authenticity Score.

Combines AI detection + EXIF/device metadata + watermark + C2PA provenance
+ cross-modal consistency into one explainable [0,1] score.

Signal weights:
  AI detection      60%  — core calibrated signal
  Metadata signals  15%  — EXIF, device fingerprint
  Watermark         10%  — strong when present
  Provenance (C2PA)  8%  — strongest human-origin signal when verified
  Cross-modal        7%  — consistency bonus for multi-modal content

Signals are applied as signed adjustments on top of the base AI score
so the underlying detector calibration is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

AI_LABEL_HIGH   = 0.75
AI_LABEL_LOW    = 0.40
CONFIDENCE_HIGH = 0.80
CONFIDENCE_MED  = 0.50

WEIGHTS = {
    "ai_detection": 0.60,
    "metadata":     0.15,
    "watermark":    0.10,
    "provenance":   0.08,
    "cross_modal":  0.07,
}


@dataclass
class SignalContribution:
    name:        str
    raw_value:   float
    contribution: float
    weight:      float
    direction:   str       # "towards_ai" | "towards_human" | "neutral"
    description: str


@dataclass
class AuthenticityScore:
    score:               float
    label:               str
    confidence:          float
    confidence_level:    str

    ai_detection_score:  float
    metadata_score:      float
    watermark_score:     float
    provenance_score:    float
    cross_modal_score:   float

    total_adjustment:    float
    signal_contributions: list[SignalContribution]

    content_hash:        str | None
    hash_verified:       bool
    c2pa_verified:       bool
    provenance_chain:    list[dict]

    top_evidence:        list[dict]
    verdict_explanation: str


def compute_authenticity_score(
    ai_score:         float,
    content_hash:     str | None           = None,
    stored_hash:      str | None           = None,
    metadata_signals: dict[str, Any] | None = None,
    watermark_result: dict[str, Any] | None = None,
    provenance_data:  dict[str, Any] | None = None,
    cross_modal_data: dict[str, Any] | None = None,
    content_type:     str                   = "text",
) -> AuthenticityScore:
    meta  = metadata_signals or {}
    wm    = watermark_result  or {}
    prov  = provenance_data   or {}
    cm    = cross_modal_data  or {}

    contributions: list[SignalContribution] = []

    contributions.append(SignalContribution(
        name="ai_detection", raw_value=ai_score, contribution=0.0,
        weight=WEIGHTS["ai_detection"],
        direction="towards_ai" if ai_score >= 0.5 else "towards_human",
        description=f"AI detection ensemble score: {ai_score:.1%}",
    ))

    # Metadata
    meta_adj = _metadata_adj(meta, content_type)
    if abs(meta_adj) > 0.001:
        contributions.append(SignalContribution(
            name="metadata_signals", raw_value=abs(meta_adj) / 0.15,
            contribution=meta_adj, weight=WEIGHTS["metadata"],
            direction="towards_ai" if meta_adj > 0 else "towards_human",
            description=_meta_desc(meta, meta_adj),
        ))

    # Watermark
    wm_adj = _watermark_adj(wm)
    if abs(wm_adj) > 0.001:
        contributions.append(SignalContribution(
            name="watermark", raw_value=float(wm.get("confidence", 0.0)),
            contribution=wm_adj, weight=WEIGHTS["watermark"],
            direction="towards_ai" if wm_adj > 0 else "towards_human",
            description=_wm_desc(wm),
        ))

    # Provenance
    prov_adj, c2pa_verified, chain = _provenance_adj(prov)
    if abs(prov_adj) > 0.001:
        contributions.append(SignalContribution(
            name="provenance", raw_value=abs(prov_adj) / 0.08,
            contribution=prov_adj, weight=WEIGHTS["provenance"],
            direction="towards_ai" if prov_adj > 0 else "towards_human",
            description=_prov_desc(prov, c2pa_verified),
        ))

    # Cross-modal
    cm_adj = _cross_modal_adj(cm)
    if abs(cm_adj) > 0.001:
        contributions.append(SignalContribution(
            name="cross_modal", raw_value=float(cm.get("inconsistency_score", 0.0)),
            contribution=cm_adj, weight=WEIGHTS["cross_modal"],
            direction="towards_ai" if cm_adj > 0 else "towards_human",
            description=_cm_desc(cm),
        ))

    total_adj   = sum(c.contribution for c in contributions if c.name != "ai_detection")
    final_score = float(np.clip(ai_score + total_adj, 0.01, 0.99))

    label = (
        "AI"        if final_score >= AI_LABEL_HIGH else
        "HUMAN"     if final_score <= AI_LABEL_LOW  else
        "UNCERTAIN"
    )
    confidence = abs(final_score - 0.5) * 2.0
    conf_level = (
        "high"   if confidence >= CONFIDENCE_HIGH else
        "medium" if confidence >= CONFIDENCE_MED  else
        "low"
    )

    hash_verified = (
        content_hash is not None
        and stored_hash is not None
        and content_hash == stored_hash
    )

    meta_score = float(np.clip(0.5 + meta_adj  / 0.30, 0.0, 1.0))
    wm_score   = float(np.clip(0.5 + wm_adj    / 0.20, 0.0, 1.0))
    prov_score = float(np.clip(0.5 - prov_adj  / 0.16, 0.0, 1.0))
    cm_score   = float(np.clip(0.5 + cm_adj    / 0.14, 0.0, 1.0))

    top_evidence = _top_evidence(contributions, wm, c2pa_verified, chain, cm, hash_verified)
    verdict      = _verdict(label, final_score, contributions, c2pa_verified)

    return AuthenticityScore(
        score=round(final_score, 4), label=label,
        confidence=round(confidence, 4), confidence_level=conf_level,
        ai_detection_score=round(ai_score, 4),
        metadata_score=round(meta_score, 4),
        watermark_score=round(wm_score, 4),
        provenance_score=round(prov_score, 4),
        cross_modal_score=round(cm_score, 4),
        total_adjustment=round(total_adj, 4),
        signal_contributions=contributions,
        content_hash=content_hash,
        hash_verified=hash_verified,
        c2pa_verified=c2pa_verified,
        provenance_chain=chain,
        top_evidence=top_evidence,
        verdict_explanation=verdict,
    )


# ── Adjusters ─────────────────────────────────────────────────

def _metadata_adj(meta: dict, content_type: str) -> float:
    adj = 0.0
    dev  = meta.get("device_fingerprint", {})
    exif = meta.get("exif", {})
    if dev.get("likely_ai_generated"):
        adj += 0.08 * min(len(dev.get("suspicious_signals", [])) + 1, 3) / 3
    if dev.get("likely_camera_capture") and content_type == "image":
        adj -= 0.06
    sw = str(exif.get("image_software", "")).lower()
    if any(t in sw for t in ["stable diffusion", "dall", "midjourney", "firefly",
                               "imagen", "flux", "automatic1111", "comfy"]):
        adj += 0.12
    if not exif.get("has_exif") and content_type == "image":
        adj += 0.04
    return float(np.clip(adj, -0.15, 0.15))


def _watermark_adj(wm: dict) -> float:
    if not wm.get("watermark_detected"):
        return 0.0
    conf = float(wm.get("confidence", 0.0))
    return round(0.10 * conf if conf > 0.70 else 0.05 * conf, 4)


def _provenance_adj(prov: dict) -> tuple[float, bool, list]:
    c2pa  = bool(prov.get("c2pa_verified", False))
    chain = list(prov.get("provenance_chain", []))
    adj   = -0.20 if c2pa else 0.0
    if prov.get("trusted_publisher"):
        adj -= 0.05
    return float(np.clip(adj, -0.20, 0.08)), c2pa, chain


def _cross_modal_adj(cm: dict) -> float:
    incon = float(cm.get("inconsistency_score", 0.0))
    n_mm  = len(cm.get("top_mismatches", []))
    if incon > 0.3 or n_mm > 0:
        return round(min(incon * 0.10 + n_mm * 0.02, 0.07), 4)
    return 0.0


# ── Description helpers ────────────────────────────────────────

def _meta_desc(meta: dict, adj: float) -> str:
    if adj > 0:
        sigs = meta.get("device_fingerprint", {}).get("suspicious_signals", [])
        return (f"Suspicious metadata signals: {', '.join(str(s) for s in sigs[:2])}"
                if sigs else "Missing or anomalous metadata for claimed source")
    return "EXIF metadata consistent with authentic capture device"


def _wm_desc(wm: dict) -> str:
    wt   = wm.get("watermark_type", "unknown")
    conf = float(wm.get("confidence", 0.0))
    z    = wm.get("z_score")
    z_s  = f" (z={z:.2f})" if z is not None else ""
    return f"{wt} watermark detected{z_s} — confidence {conf:.0%}"


def _prov_desc(prov: dict, c2pa: bool) -> str:
    if c2pa:
        issuer = prov.get("c2pa_issuer", "trusted authority")
        return f"C2PA provenance verified by {issuer}"
    return "Provenance metadata present but not cryptographically verified"


def _cm_desc(cm: dict) -> str:
    mm = cm.get("top_mismatches", [])
    if mm:
        names = [m["check"].replace("_", " ") for m in mm[:2]]
        return f"Cross-modal inconsistency: {', '.join(names)}"
    return f"Cross-modal inconsistency score: {cm.get('inconsistency_score', 0):.2f}"


def _top_evidence(contributions, wm, c2pa, chain, cm, hash_ok) -> list[dict]:
    ev: list[dict] = []
    for c in contributions:
        if abs(c.contribution) > 0.02 or c.name == "ai_detection":
            sev = ("high" if abs(c.contribution) >= 0.08 else
                   "medium" if abs(c.contribution) >= 0.03 else "low")
            ev.append({"signal": c.name.replace("_", " ").title(),
                        "value": f"{c.raw_value:.1%}", "adjustment": f"{c.contribution:+.3f}",
                        "direction": c.direction, "description": c.description,
                        "severity": sev})
    if c2pa:
        ev.append({"signal": "C2PA Provenance", "value": "verified",
                    "adjustment": "−0.200", "direction": "towards_human",
                    "description": "Cryptographic content authenticity verified",
                    "severity": "high"})
    if hash_ok:
        ev.append({"signal": "Integrity Check", "value": "passed",
                    "adjustment": "+0.000", "direction": "neutral",
                    "description": "SHA-256 content hash matches stored value",
                    "severity": "low"})
    return ev[:8]


def _verdict(label, score, contributions, c2pa) -> str:
    pct = round(score * 100)
    if c2pa and label == "HUMAN":
        return (f"This content has a cryptographically verified origin (C2PA) "
                f"and scores {pct}% AI probability — consistent with authentic content.")
    if label == "AI":
        active = [c for c in contributions
                  if c.direction == "towards_ai" and c.name != "ai_detection"]
        ss = (f" Corroborating signals: {', '.join(c.name.replace('_',' ') for c in active[:2])}."
              if active else "")
        return f"AI-generated content detected with {pct}% probability.{ss}"
    if label == "HUMAN":
        return (f"This content appears authentic with {100-pct}% human probability. "
                f"No significant AI generation artifacts detected.")
    return (f"Inconclusive — AI probability is {pct}%, in the uncertain range (40–75%). "
            f"Manual review recommended for high-stakes decisions.")
