"""
Step 80: Ensemble meta-classifier that combines all detector outputs.

When multiple detectors run on the same content (e.g. a video has both
video deepfake detection AND audio deepfake detection), their outputs
are fed into this meta-classifier to produce the final unified score.

Architecture:
  Input:  DetectorOutput objects from all applicable detectors
  Model:  XGBoost gradient-boosted classifier
  Output: Calibrated [0, 1] AI probability + label

Feature vector per job:
  - Per-detector scores (one slot per detector, 0.5 if unavailable)
  - Per-detector confidence
  - Per-detector error flag (1 = errored)
  - Cross-detector agreement std (high disagreement = uncertain)
  - Content-type one-hot encoding
  - Metadata signals (EXIF anomaly, watermark detection, C2PA)
  - Total n_flagged_segments (video/audio)
  - Processing time z-score (unusually fast/slow = suspicious)

Total: ~25 features
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from .dispatcher import DetectorOutput, ContentType

log = structlog.get_logger(__name__)

CONTENT_TYPE_IDX = {"text": 0, "image": 1, "video": 2, "audio": 3, "code": 4}
N_DETECTORS = 5

FEATURE_NAMES = [
    # Per-detector scores (5 slots — 0.5 if unavailable)
    "score_text", "score_image", "score_video", "score_audio", "score_code",
    # Per-detector confidence
    "conf_text", "conf_image", "conf_video", "conf_audio", "conf_code",
    # Per-detector error flag
    "err_text", "err_image", "err_video", "err_audio", "err_code",
    # Ensemble statistics
    "score_mean", "score_max", "score_min", "score_std",
    # Metadata signals
    "watermark_detected", "exif_ai_signal", "c2pa_verified",
    # Content-type one-hot (5)
    "is_text", "is_image", "is_video", "is_audio", "is_code",
    # Segment signals
    "n_flagged_segments", "max_segment_score",
]

N_FEATURES = len(FEATURE_NAMES)


def build_multi_detector_feature_vector(
    outputs: list[DetectorOutput],
    metadata_signals: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Construct the feature vector from one or more DetectorOutputs.
    Handles partial inputs gracefully (e.g. only video was run).
    """
    meta = metadata_signals or {}

    # Index outputs by content_type
    by_type: dict[str, DetectorOutput] = {}
    for out in outputs:
        by_type[out.content_type] = out

    # Determine the primary content_type (first output, or majority)
    primary_type = outputs[0].content_type if outputs else "text"

    # Per-detector scores and confidence
    scores = []
    confs  = []
    errors = []
    for ct in ["text", "image", "video", "audio", "code"]:
        out = by_type.get(ct)
        scores.append(out.score      if out and not out.error else 0.5)
        confs.append(out.confidence  if out and not out.error else 0.0)
        errors.append(1.0 if (out and out.error) else 0.0)

    # Ensemble statistics (only over available detectors)
    available_scores = [s for s, e in zip(scores, errors) if e == 0.0]
    if available_scores:
        score_mean = float(np.mean(available_scores))
        score_max  = float(max(available_scores))
        score_min  = float(min(available_scores))
        score_std  = float(np.std(available_scores))
    else:
        score_mean = score_max = score_min = 0.5
        score_std  = 0.0

    # Metadata signals
    wm  = meta.get("watermark", {})
    dev = meta.get("device_fingerprint", {})
    pro = meta.get("provenance", {})
    watermark_detected = float(wm.get("watermark_detected", False))
    exif_ai_signal     = float(dev.get("likely_ai_generated", False))
    c2pa_verified      = float(pro.get("c2pa_verified", False))

    # Content-type one-hot
    ct_onehot = [0.0] * N_DETECTORS
    ct_idx = CONTENT_TYPE_IDX.get(primary_type, 0)
    ct_onehot[ct_idx] = 1.0

    # Segment signals (video/audio)
    all_flagged: list[dict] = []
    for out in outputs:
        all_flagged.extend(out.flagged_segments)
    n_flagged    = float(len(all_flagged))
    max_seg_score = float(max((s.get("score", 0.0) for s in all_flagged), default=0.0))

    vec = np.array(
        scores
        + confs
        + errors
        + [score_mean, score_max, score_min, score_std]
        + [watermark_detected, exif_ai_signal, c2pa_verified]
        + ct_onehot
        + [n_flagged, max_seg_score],
        dtype=np.float32,
    )

    assert len(vec) == N_FEATURES, f"Expected {N_FEATURES}, got {len(vec)}"
    return vec


@dataclass
class EnsembleOutput:
    """Final output from the ensemble meta-classifier."""
    score:              float
    label:              str
    confidence:         float
    feature_vector:     np.ndarray
    contributing_detectors: list[str]
    per_detector_scores: dict[str, float]
    metadata_adjustment: float


class EnsembleMetaClassifier:
    """
    Multi-detector ensemble meta-classifier.
    Combines outputs from all available detectors for a single job.

    Falls back to a weighted average when the XGBoost model is unavailable
    (before training completes).
    """

    # Fallback weights per detector type (based on typical accuracy)
    FALLBACK_WEIGHTS: dict[str, float] = {
        "text":  0.35,
        "image": 0.25,
        "video": 0.20,
        "audio": 0.15,
        "code":  0.05,
    }

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self._xgb:      Any = None
        self._platt:    Any = None
        self._isotonic: Any = None
        self._is_fitted = False
        self._checkpoint = checkpoint_path

    def load(self) -> None:
        if not self._checkpoint or not self._checkpoint.exists():
            log.info("ensemble_meta_no_checkpoint — using weighted average fallback")
            return
        try:
            with (self._checkpoint / "xgb.pkl").open("rb") as f:
                self._xgb = pickle.load(f)
            with (self._checkpoint / "platt.pkl").open("rb") as f:
                self._platt = pickle.load(f)
            with (self._checkpoint / "isotonic.pkl").open("rb") as f:
                self._isotonic = pickle.load(f)
            self._is_fitted = True
            log.info("ensemble_meta_loaded")
        except Exception as exc:
            log.warning("ensemble_meta_load_failed", error=str(exc))

    def predict(
        self,
        outputs: list[DetectorOutput],
        metadata_signals: dict[str, Any] | None = None,
    ) -> EnsembleOutput:
        """
        Combine all detector outputs into a single calibrated score.
        """
        if not outputs:
            return self._empty_output()

        fv = build_multi_detector_feature_vector(outputs, metadata_signals)

        if self._is_fitted:
            score = self._predict_xgb(fv)
        else:
            score = self._weighted_average(outputs)

        # Apply metadata adjustments (same logic as backend result_engine)
        score, adjustment = self._apply_metadata_adjustment(score, metadata_signals or {})

        label = (
            "AI"        if score >= 0.75 else
            "HUMAN"     if score <= 0.40 else
            "UNCERTAIN"
        )
        confidence = round(abs(score - 0.5) * 2, 4)

        contributing = [o.content_type for o in outputs if not o.error]
        per_detector  = {o.content_type: o.score for o in outputs}

        return EnsembleOutput(
            score=round(score, 4),
            label=label,
            confidence=confidence,
            feature_vector=fv,
            contributing_detectors=contributing,
            per_detector_scores=per_detector,
            metadata_adjustment=round(adjustment, 4),
        )

    def _predict_xgb(self, fv: np.ndarray) -> float:
        import numpy as np
        X = fv.reshape(1, -1)
        raw = float(self._xgb.predict_proba(X)[0, 1])

        if self._platt and self._isotonic:
            raw_clipped = max(1e-6, min(1 - 1e-6, raw))
            import math
            log_odds = math.log(raw_clipped / (1 - raw_clipped))
            platt_p  = float(self._platt.predict_proba([[log_odds]])[0, 1])
            iso_p    = float(self._isotonic.predict([raw])[0])
            return float(np.clip((platt_p + iso_p) / 2.0, 0.01, 0.99))
        return float(np.clip(raw, 0.01, 0.99))

    def _weighted_average(self, outputs: list[DetectorOutput]) -> float:
        """Fallback: weighted average over available (non-errored) detectors."""
        total_weight = 0.0
        weighted_sum = 0.0
        for out in outputs:
            if out.error:
                continue
            w = self.FALLBACK_WEIGHTS.get(out.content_type, 0.1)
            weighted_sum  += out.score * w
            total_weight  += w
        if total_weight == 0:
            return 0.5
        return float(np.clip(weighted_sum / total_weight, 0.01, 0.99))

    @staticmethod
    def _apply_metadata_adjustment(
        score: float,
        metadata: dict[str, Any],
    ) -> tuple[float, float]:
        adjustment = 0.0
        wm  = metadata.get("watermark", {})
        dev = metadata.get("device_fingerprint", {})
        pro = metadata.get("provenance", {})

        if wm.get("watermark_detected") and wm.get("confidence", 0) > 0.7:
            adjustment += 0.10 * float(wm["confidence"])
        if dev.get("likely_ai_generated"):
            adjustment += 0.06
        if dev.get("likely_camera_capture"):
            adjustment -= 0.04
        if pro.get("c2pa_verified"):
            adjustment -= 0.15

        final = float(np.clip(score + adjustment, 0.01, 0.99))
        return final, adjustment

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        output_dir: Path | None = None,
    ) -> None:
        """Train the XGBoost meta-classifier on labelled detector outputs."""
        from xgboost import XGBClassifier               # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.isotonic import IsotonicRegression      # type: ignore

        log.info("fitting_ensemble_meta", n_train=len(X_train), n_val=len(X_val))

        self._xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            early_stopping_rounds=20, n_jobs=-1, random_state=42,
        )
        self._xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=25)

        raw_probs = self._xgb.predict_proba(X_val)[:, 1]
        raw_clipped = np.clip(raw_probs, 1e-6, 1 - 1e-6)
        import math
        log_odds = np.array([math.log(p / (1 - p)) for p in raw_clipped]).reshape(-1, 1)

        self._platt = LogisticRegression(C=1e4)
        self._platt.fit(log_odds, y_val)
        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(raw_probs, y_val)
        self._is_fitted = True

        if output_dir:
            self._save(output_dir)
        log.info("ensemble_meta_fitted")

    def _save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, obj in [("xgb", self._xgb), ("platt", self._platt),
                           ("isotonic", self._isotonic)]:
            with (directory / f"{name}.pkl").open("wb") as f:
                pickle.dump(obj, f)
        meta = {"n_features": N_FEATURES, "feature_names": FEATURE_NAMES}
        with (directory / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)
        log.info("ensemble_meta_saved", directory=str(directory))

    @staticmethod
    def _empty_output() -> EnsembleOutput:
        return EnsembleOutput(
            score=0.5, label="UNCERTAIN", confidence=0.0,
            feature_vector=np.zeros(N_FEATURES, dtype=np.float32),
            contributing_detectors=[], per_detector_scores={},
            metadata_adjustment=0.0,
        )
