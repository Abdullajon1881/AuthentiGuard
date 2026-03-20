"""
Steps 121–124: Authenticity Passport.

Step 121: Assign cryptographic signatures to every analyzed file.
Step 122: Build content hash + origin metadata + authenticity score
          + full verification history per file.
Step 123: Build provenance chain tracking: creator, timestamps,
          modifications, detection results at each stage.
Step 124: Public verification API so third parties can verify
          any file's Authenticity Passport.

An Authenticity Passport is a permanent, tamper-evident record
attached to every file analysed by AuthentiGuard:

  {
    "passport_id":     "uuid",
    "content_hash":    "sha256:abc123...",
    "issued_at":       "2024-03-15T12:00:00Z",
    "issuer":          "AuthentiGuard v0.1.0",
    "authenticity_score": 0.23,
    "label":           "HUMAN",
    "provenance_chain": [...],
    "verification_history": [...],
    "signature":       "ECDSA P-256 signature"
  }

Passports are:
  - Stored in S3 (reports bucket) indexed by content_hash
  - Retrievable via GET /api/v1/passport/{content_hash}
  - Verifiable without an AuthentiGuard account (public endpoint)
  - Optionally anchored to a blockchain (Step 139)
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

ISSUER_NAME    = "AuthentiGuard"
ISSUER_VERSION = "0.1.0"


# ── Data structures ───────────────────────────────────────────

@dataclass
class ProvenanceEntry:
    """One entry in the provenance chain."""
    stage:       str        # "capture" | "upload" | "edit" | "analysis" | "distribution"
    actor:       str        # who performed this action
    tool:        str | None # software used
    timestamp:   str        # ISO 8601
    content_hash: str       # SHA-256 at this stage
    notes:       str | None = None


@dataclass
class VerificationRecord:
    """One verification event (third-party checked the passport)."""
    verified_at:  str
    verifier_ip:  str | None
    result:       str         # "valid" | "invalid" | "hash_mismatch"
    check_type:   str         # "signature" | "hash" | "full"


@dataclass
class AuthenticityPassport:
    """
    The complete Authenticity Passport for one piece of content.
    This is the canonical format stored and returned by the public API.
    """
    passport_id:          str
    content_hash:         str       # sha256:<hex>
    content_type:         str       # "text" | "image" | "video" | "audio" | "code"
    filename:             str
    file_size_bytes:      int

    # Detection result
    authenticity_score:   float
    label:                str       # "AI" | "HUMAN" | "UNCERTAIN"
    confidence:           float
    model_attribution:    dict[str, float]

    # Provenance
    provenance_chain:     list[ProvenanceEntry]
    c2pa_verified:        bool
    c2pa_issuer:          str | None

    # Issuance
    issued_at:            str       # ISO 8601
    issuer:               str       # "AuthentiGuard v0.1.0"
    job_id:               str
    report_id:            str | None

    # Integrity
    passport_hash:        str       # SHA-256 of the passport JSON (before signature)
    signature:            str       # ECDSA P-256 or HMAC-SHA256

    # Verification history (append-only)
    verification_history: list[VerificationRecord] = field(default_factory=list)

    # Optional blockchain anchor (Step 139)
    blockchain_tx:        str | None = None


# ── Passport issuer ───────────────────────────────────────────

class PassportIssuer:
    """
    Issues and stores Authenticity Passports.
    Uses ECDSA P-256 for signatures when available, HMAC-SHA256 otherwise.
    """

    def __init__(
        self,
        signing_key: bytes | str,
        use_ecdsa: bool = True,
    ) -> None:
        self._signing_key = signing_key
        self._use_ecdsa   = use_ecdsa
        self._ecdsa_signer: Any = None

        if use_ecdsa:
            try:
                from cryptography.hazmat.primitives.asymmetric import ec      # type: ignore
                from cryptography.hazmat.primitives import serialization       # type: ignore
                if isinstance(signing_key, bytes):
                    self._ecdsa_signer = serialization.load_pem_private_key(
                        signing_key, password=None
                    )
                log.info("passport_issuer_ecdsa_ready")
            except Exception as exc:
                log.warning("ecdsa_unavailable", error=str(exc))
                self._use_ecdsa = False

    def issue(
        self,
        content_bytes:      bytes,
        filename:           str,
        content_type:       str,
        detection_result:   dict[str, Any],
        provenance_signals: dict[str, Any] | None = None,
        job_id:             str | None = None,
        report_id:          str | None = None,
    ) -> AuthenticityPassport:
        """
        Issue an Authenticity Passport for a piece of content.
        """
        prov      = provenance_signals or {}
        job_id    = job_id    or str(uuid.uuid4())
        report_id = report_id or None

        content_hash = f"sha256:{hashlib.sha256(content_bytes).hexdigest()}"
        passport_id  = str(uuid.uuid4())
        now          = datetime.now(timezone.utc).isoformat()

        # Build provenance chain
        chain = self._build_chain(content_hash, filename, now, prov, detection_result)

        passport = AuthenticityPassport(
            passport_id=passport_id,
            content_hash=content_hash,
            content_type=content_type,
            filename=filename,
            file_size_bytes=len(content_bytes),
            authenticity_score=float(detection_result.get("score", 0.5)),
            label=str(detection_result.get("label", "UNCERTAIN")),
            confidence=float(detection_result.get("confidence", 0.0)),
            model_attribution=dict(detection_result.get("model_attribution", {})),
            provenance_chain=chain,
            c2pa_verified=bool(prov.get("c2pa_verified", False)),
            c2pa_issuer=prov.get("c2pa_issuer"),
            issued_at=now,
            issuer=f"{ISSUER_NAME} v{ISSUER_VERSION}",
            job_id=job_id,
            report_id=report_id,
            passport_hash="",   # filled below
            signature="",
        )

        # Sign
        passport_bytes = self._canonical_bytes(passport)
        passport.passport_hash = hashlib.sha256(passport_bytes).hexdigest()
        passport.signature     = self._sign(passport_bytes)

        log.info("passport_issued",
                  passport_id=passport_id,
                  content_hash=content_hash[:20],
                  label=passport.label,
                  score=passport.authenticity_score)
        return passport

    def _build_chain(
        self,
        content_hash: str,
        filename:     str,
        now:          str,
        prov:         dict,
        result:       dict,
    ) -> list[ProvenanceEntry]:
        chain: list[ProvenanceEntry] = []

        # Origin entry (from C2PA or metadata)
        if prov.get("c2pa_verified") and prov.get("provenance_chain"):
            for step in prov["provenance_chain"][:3]:
                chain.append(ProvenanceEntry(
                    stage=step.get("step", "unknown"),
                    actor=step.get("actor", "unknown"),
                    tool=step.get("tool"),
                    timestamp=step.get("time", now),
                    content_hash=content_hash,
                    notes=step.get("note"),
                ))
        else:
            chain.append(ProvenanceEntry(
                stage="upload",
                actor="user",
                tool="AuthentiGuard API",
                timestamp=now,
                content_hash=content_hash,
            ))

        # Analysis entry
        chain.append(ProvenanceEntry(
            stage="analysis",
            actor=f"{ISSUER_NAME} v{ISSUER_VERSION}",
            tool="Multi-modal ensemble detector",
            timestamp=now,
            content_hash=content_hash,
            notes=(
                f"AI probability: {result.get('score',0):.1%} — "
                f"Label: {result.get('label','UNCERTAIN')}"
            ),
        ))

        return chain

    def _canonical_bytes(self, passport: AuthenticityPassport) -> bytes:
        """Deterministic serialisation for signing (excludes signature field)."""
        d = asdict(passport)
        d.pop("signature", None)
        d.pop("verification_history", None)
        return json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")

    def _sign(self, data: bytes) -> str:
        if self._use_ecdsa and self._ecdsa_signer:
            from cryptography.hazmat.primitives import hashes           # type: ignore
            from cryptography.hazmat.primitives.asymmetric import ec    # type: ignore
            sig = self._ecdsa_signer.sign(data, ec.ECDSA(hashes.SHA256()))
            return base64.b64encode(sig).decode("ascii")
        # HMAC fallback
        import hmac as hmac_mod
        key = (self._signing_key if isinstance(self._signing_key, bytes)
               else self._signing_key.encode())
        return hmac_mod.new(key, data, hashlib.sha256).hexdigest()


# ── Passport verifier (Step 124: public verification) ─────────

class PassportVerifier:
    """
    Public verifier — no authentication required.
    Anyone can verify an Authenticity Passport using the public key.
    """

    def __init__(
        self,
        public_key_pem: bytes | None = None,
        hmac_key:       str | None   = None,
    ) -> None:
        self._public_key: Any = None
        self._hmac_key        = hmac_key

        if public_key_pem:
            try:
                from cryptography.hazmat.primitives.serialization import (  # type: ignore
                    load_pem_public_key
                )
                self._public_key = load_pem_public_key(public_key_pem)
                log.info("passport_verifier_ecdsa_ready")
            except Exception as exc:
                log.warning("ecdsa_public_key_failed", error=str(exc))

    def verify(self, passport: AuthenticityPassport) -> dict[str, Any]:
        """
        Verify an Authenticity Passport.
        Returns a structured verification result.
        """
        errors: list[str] = []
        checks: dict[str, bool] = {}

        # 1. Re-compute passport hash
        sig           = passport.signature
        passport.signature = ""
        canon_bytes   = self._canonical_bytes(passport)
        passport.signature = sig

        recomputed_hash = hashlib.sha256(canon_bytes).hexdigest()
        checks["hash_valid"] = recomputed_hash == passport.passport_hash
        if not checks["hash_valid"]:
            errors.append("Passport hash mismatch — passport may have been tampered with")

        # 2. Verify signature
        if self._public_key:
            checks["signature_valid"] = self._verify_ecdsa(canon_bytes, sig)
        elif self._hmac_key:
            checks["signature_valid"] = self._verify_hmac(canon_bytes, sig)
        else:
            checks["signature_valid"] = False
            errors.append("No verification key configured")

        if not checks["signature_valid"]:
            errors.append("Signature verification failed")

        # 3. Check issuer
        checks["issuer_known"] = passport.issuer.startswith(ISSUER_NAME)

        # 4. Check provenance chain integrity
        checks["chain_intact"] = len(passport.provenance_chain) >= 1
        if not checks["chain_intact"]:
            errors.append("Provenance chain is empty")

        is_valid = all(checks.values()) and not errors

        return {
            "is_valid":      is_valid,
            "passport_id":   passport.passport_id,
            "content_hash":  passport.content_hash,
            "label":         passport.label,
            "score":         passport.authenticity_score,
            "issued_at":     passport.issued_at,
            "issuer":        passport.issuer,
            "c2pa_verified": passport.c2pa_verified,
            "checks":        checks,
            "errors":        errors,
            "verified_at":   datetime.now(timezone.utc).isoformat(),
        }

    def _canonical_bytes(self, passport: AuthenticityPassport) -> bytes:
        d = asdict(passport)
        d.pop("signature", None)
        d.pop("verification_history", None)
        return json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")

    def _verify_ecdsa(self, data: bytes, sig_b64: str) -> bool:
        try:
            from cryptography.hazmat.primitives import hashes           # type: ignore
            from cryptography.hazmat.primitives.asymmetric import ec    # type: ignore
            from cryptography.exceptions import InvalidSignature        # type: ignore
            sig = base64.b64decode(sig_b64.encode())
            self._public_key.verify(sig, data, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False

    def _verify_hmac(self, data: bytes, sig: str) -> bool:
        import hmac as hmac_mod
        key = (self._hmac_key.encode() if isinstance(self._hmac_key, str)
               else self._hmac_key)
        expected = hmac_mod.new(key, data, hashlib.sha256).hexdigest()
        return hmac_mod.compare_digest(expected, sig)


# ── Public Passport registry (Step 124) ───────────────────────

class PassportRegistry:
    """
    Stores and retrieves Authenticity Passports by content hash.
    Backed by S3 in production; in-memory dict for testing.
    """

    def __init__(self, s3_client: Any = None, bucket: str = "") -> None:
        self._s3     = s3_client
        self._bucket = bucket
        self._cache: dict[str, AuthenticityPassport] = {}

    async def store(self, passport: AuthenticityPassport) -> str:
        """Store a passport. Returns the S3 key or cache key."""
        key = f"passports/{passport.content_hash.replace('sha256:', '')}.json"
        data = json.dumps(asdict(passport), indent=2, default=str).encode()

        if self._s3 and self._bucket:
            self._s3.put_object(
                Bucket=self._bucket, Key=key,
                Body=data, ContentType="application/json",
                ServerSideEncryption="AES256",
            )
        else:
            self._cache[passport.content_hash] = passport

        log.info("passport_stored", key=key)
        return key

    async def get(self, content_hash: str) -> AuthenticityPassport | None:
        """Retrieve a passport by content hash (Step 124: public lookup)."""
        if content_hash in self._cache:
            return self._cache[content_hash]

        if not self._s3:
            return None

        hash_hex = content_hash.replace("sha256:", "")
        key      = f"passports/{hash_hex}.json"
        try:
            obj  = self._s3.get_object(Bucket=self._bucket, Key=key)
            data = json.loads(obj["Body"].read())
            # Reconstruct passport from dict
            chain = [ProvenanceEntry(**e) for e in data.pop("provenance_chain", [])]
            vh    = [VerificationRecord(**v) for v in data.pop("verification_history", [])]
            return AuthenticityPassport(**data, provenance_chain=chain,
                                        verification_history=vh)
        except Exception as exc:
            log.debug("passport_not_found", hash=hash_hex[:16], error=str(exc))
            return None

    async def record_verification(
        self,
        content_hash: str,
        result:       str,
        verifier_ip:  str | None = None,
    ) -> None:
        """Append a verification record (Step 124: public verification tracking)."""
        passport = await self.get(content_hash)
        if not passport:
            return
        passport.verification_history.append(VerificationRecord(
            verified_at=datetime.now(timezone.utc).isoformat(),
            verifier_ip=verifier_ip,
            result=result,
            check_type="full",
        ))
        await self.store(passport)
