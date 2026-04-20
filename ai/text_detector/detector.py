"""
AuthentiGuard text detector — orchestrates L1 + L2 + L3 + meta-classifier.

Reliability-gated 3-zone decision policy:
    score >= 0.70  →  AI
    score <= 0.30  →  HUMAN
    else           →  UNCERTAIN

One gating rule:
    G1  short text (<50 words) → UNCERTAIN (insufficient signal for L1/L2)

(G2, the L2–L3 disagreement gate, was removed in 3.2 because |L2−L3| mean is
0.404 — it fired on ~48% of inputs and collapsed AI recall to 2.3%.)

Score selection (first match wins):
    1. Stage 2 LR + isotonic calibrator        (checkpoints/meta_*.joblib)
    2. Legacy XGB MetaClassifier                (if explicitly passed — unused)
    3. Stage 1 fixed-weight combiner [0.20, 0.35, 0.45]   (authoritative fallback)
"""

from __future__ import annotations

from dataclasses import dataclass  # noqa: F401
from pathlib import Path
from typing import Any

import structlog

from .layers import (
    LayerResult,
    PerplexityLayer,
    SemanticLayer,
    StylometryLayer,
)
from .meta import EnsembleResult, MetaClassifier, build_feature_vector

log = structlog.get_logger(__name__)

# Production model-version string. Bumped whenever the inference stack
# changes in a way that affects score semantics. Consumed by
# backend/app/observability/prediction_log.py.
MODEL_VERSION = "3.2-g2-removed-product-output"

# Decision thresholds — single source of truth.
ZONE_AI_THRESHOLD     = 0.70
ZONE_HUMAN_THRESHOLD  = 0.30
SHORT_TEXT_WORD_MIN   = 50

# Confidence bands (score-distance-from-0.5 scaled to [0, 1]).
BAND_HIGH   = 0.60
BAND_MEDIUM = 0.30


def _score_to_confidence(score: float) -> float:
    """Convert [0, 1] probability to a [0, 1] confidence (distance from 0.5)."""
    return round(abs(score - 0.5) * 2, 4)


class TextDetector:
    """
    Main text detection ensemble.

    load_models() is called once at worker startup (heavy — loads GPT-2 + DeBERTa).
    analyze()     is called per-request (~200–500ms).
    """

    def __init__(
        self,
        transformer_checkpoint: Path | None = None,
        meta_checkpoint: Path | None = None,
        device: str | None = None,
        meta_lr_path: Path | None = None,
        meta_calibrator_path: Path | None = None,
    ) -> None:
        self._layer1 = PerplexityLayer(device=device)
        self._layer2 = StylometryLayer(use_spacy=True)
        self._transformer_checkpoint = transformer_checkpoint
        self._layer3: SemanticLayer | None = None
        self._meta = MetaClassifier()
        self._meta_checkpoint = meta_checkpoint
        self._loaded = False
        self._active_layers: list[int] = []
        self._device = device

        # Stage 2: learned LR + isotonic calibration. Optional; auto-discovered
        # from the transformer checkpoint's grandparent dir if not passed.
        self._meta_lr_path = meta_lr_path
        self._meta_calibrator_path = meta_calibrator_path
        self._lr_meta = None
        self._lr_calibrator = None
        self._lr_threshold: float = 0.5
        self._lr_feature_order: list[str] = ["l1_score", "l2_score", "l3_score"]

    # ── Startup ───────────────────────────────────────────────

    def load_models(self) -> None:
        log.info("loading_text_detector_models")

        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # L1 (perplexity) and L2 (stylometry) always load — no training needed
        self._layer1.load_model()
        self._layer2.load_model()
        self._active_layers = [0, 1]

        # L3 (DeBERTa) — only load if fine-tuned checkpoint exists
        if self._transformer_checkpoint and self._transformer_checkpoint.exists():
            self._layer3 = SemanticLayer(
                checkpoint_path=self._transformer_checkpoint,
                device=self._device,
            )
            self._layer3.load_model()
            self._active_layers.append(2)
            log.info("layer3_loaded", checkpoint=str(self._transformer_checkpoint))
        else:
            log.warning(
                "layer3_skipped",
                reason="no fine-tuned checkpoint at "
                       + str(self._transformer_checkpoint or "None"),
            )

        # Legacy XGB meta (unused in prod) — only load if explicitly passed.
        if self._meta_checkpoint and self._meta_checkpoint.exists():
            self._meta = MetaClassifier.load(self._meta_checkpoint)
        else:
            log.info("legacy_xgb_meta_not_loaded — using stage-2 lr or fixed weights")

        # ── Stage 2: LR + isotonic calibrator ────────────────
        # Auto-discover checkpoints/meta_classifier.joblib + meta_calibrator.joblib
        # next to the transformer checkpoint (both live under checkpoints/).
        if (
            self._meta_lr_path is None
            and self._meta_calibrator_path is None
            and self._transformer_checkpoint is not None
        ):
            try:
                base = self._transformer_checkpoint.parent.parent
                cand_lr  = base / "meta_classifier.joblib"
                cand_cal = base / "meta_calibrator.joblib"
                if cand_lr.exists() and cand_cal.exists():
                    self._meta_lr_path = cand_lr
                    self._meta_calibrator_path = cand_cal
                    log.info(
                        "lr_meta_auto_discovered",
                        lr_path=str(cand_lr),
                        calibrator_path=str(cand_cal),
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("lr_meta_auto_discovery_failed", error=str(exc))

        if (
            self._meta_lr_path is not None
            and self._meta_calibrator_path is not None
            and self._meta_lr_path.exists()
            and self._meta_calibrator_path.exists()
        ):
            try:
                import joblib
                lr_obj     = joblib.load(self._meta_lr_path)
                cal_bundle = joblib.load(self._meta_calibrator_path)
                if isinstance(cal_bundle, dict) and "calibrator" in cal_bundle:
                    self._lr_meta         = lr_obj
                    self._lr_calibrator   = cal_bundle["calibrator"]
                    self._lr_threshold    = float(cal_bundle.get("threshold", 0.5))
                    self._lr_feature_order = list(
                        cal_bundle.get("feature_order", self._lr_feature_order)
                    )
                    log.info(
                        "lr_meta_loaded",
                        lr_path=str(self._meta_lr_path),
                        calibrator_path=str(self._meta_calibrator_path),
                        threshold=self._lr_threshold,
                        feature_order=self._lr_feature_order,
                    )
                    self._set_meta_fallback_metric(0)
                else:
                    log.warning("lr_meta_calibrator_bundle_malformed — falling back")
                    self._lr_meta = None
                    self._lr_calibrator = None
            except Exception as exc:  # noqa: BLE001
                log.error("lr_meta_load_failed", error=str(exc))
                self._lr_meta = None
                self._lr_calibrator = None
                self._set_meta_fallback_metric(1)
        else:
            log.warning(
                "META_FALLBACK_ACTIVE",
                reason="meta_classifier.joblib / meta_calibrator.joblib not on disk; "
                       "using Stage 1 fixed-weight combiner. Calibrated probabilities "
                       "are NOT available in this mode.",
            )
            self._set_meta_fallback_metric(1)

        log.info(
            "text_detector_ready",
            active_layers=len(self._active_layers),
            layers=self._active_layers,
        )
        if len(self._active_layers) < 3:
            log.info("running_in_mvp_mode — L1+L2 only. Fine-tune L3 for full accuracy.")

        self._loaded = True

    @staticmethod
    def _set_meta_fallback_metric(value: int) -> None:
        try:
            from backend.app.core.metrics import META_CLASSIFIER_FALLBACK
            META_CLASSIFIER_FALLBACK.set(value)
        except Exception:  # noqa: BLE001
            pass

    # ── Per-request ──────────────────────────────────────────

    def analyze(self, text: str) -> EnsembleResult:
        """Run the full ensemble on `text` and return an EnsembleResult."""
        if not self._loaded:
            raise RuntimeError("Call load_models() first.")

        layer_results: list[LayerResult] = [
            self._layer1.analyze_safe(text),
            self._layer2.analyze_safe(text),
        ]
        if self._layer3 is not None:
            layer_results.append(self._layer3.analyze_safe(text))

        feature_vector = build_feature_vector(layer_results, text)

        # Score selection: LR meta > XGB meta > fixed weights
        used_lr_meta = False
        score: float
        if self._lr_meta is not None and self._lr_calibrator is not None:
            by_name = {r.layer_name: r for r in layer_results}

            def _layer_score(layer_name: str) -> float:
                r = by_name.get(layer_name)
                return float(r.score) if (r is not None and not r.error) else 0.5

            feature_map = {
                "l1_score": _layer_score("perplexity"),
                "l2_score": _layer_score("stylometry"),
                "l3_score": _layer_score("transformer"),
            }
            try:
                import numpy as _np
                X_row = _np.array(
                    [[feature_map[f] for f in self._lr_feature_order]],
                    dtype=_np.float64,
                )
                score = float(self._lr_calibrator.predict_proba(X_row)[0, 1])
                score = max(0.01, min(0.99, score))
                used_lr_meta = True
            except Exception as exc:  # noqa: BLE001
                log.error("lr_meta_inference_failed_falling_back", error=str(exc))
                used_lr_meta = False

        if not used_lr_meta:
            if self._meta._is_fitted:
                score = self._meta.predict(feature_vector)
            else:
                # Stage 1 fallback: weights from scripts/fit_ensemble_weights.py
                #   val F1 at [0.20, 0.35, 0.45] + threshold 0.41: 0.99692
                #   test F1 verify: 0.99447
                # DO NOT edit by hand — re-run the fit script in lockstep with
                # fit_weights.json.
                active_count = len(layer_results)
                if active_count == 2:
                    weights = [0.50, 0.50]
                elif active_count == 3:
                    weights = [0.20, 0.35, 0.45]
                else:
                    weights = [0.20, 0.20, 0.35, 0.25]
                scores = [r.score for r in layer_results]
                score = max(0.01, min(0.99, sum(s * w for s, w in zip(scores, weights))))

        # ── Reliability-gated decision ──────────────────────
        if score >= ZONE_AI_THRESHOLD:
            label = "AI"
        elif score <= ZONE_HUMAN_THRESHOLD:
            label = "HUMAN"
        else:
            label = "UNCERTAIN"

        # G1: short-text gate
        word_count = len(text.split())
        if word_count < SHORT_TEXT_WORD_MIN and label != "UNCERTAIN":
            label = "UNCERTAIN"

        confidence = _score_to_confidence(score)

        # ── Evidence summary for the UI ─────────────────────
        by_name = {r.layer_name: r for r in layer_results}
        evidence_summary: dict[str, Any] = {
            "layer_scores": {
                "perplexity":  by_name.get("perplexity",  LayerResult("perplexity",  0.5)).score,
                "stylometry":  by_name.get("stylometry",  LayerResult("stylometry",  0.5)).score,
                "transformer": by_name.get("transformer", LayerResult("transformer", 0.5)).score
                                if "transformer" in by_name else None,
                "adversarial": None,  # L4 not wired in production
            },
            "layer_errors": {r.layer_name: r.error for r in layer_results if r.error},
            "top_signals":      _build_top_signals(layer_results),
            "sentence_scores":  _merge_sentence_scores(layer_results),
        }

        # Product-facing block (stable, self-documenting schema).
        if confidence >= BAND_HIGH:
            confidence_band = "high"
        elif confidence >= BAND_MEDIUM:
            confidence_band = "medium"
        else:
            confidence_band = "low"

        l1_val = round(float(by_name.get("perplexity", LayerResult("perplexity", 0.5)).score), 4)
        l2_val = round(float(by_name.get("stylometry", LayerResult("stylometry", 0.5)).score), 4)
        l3_val = (
            round(float(by_name.get("transformer", LayerResult("transformer", 0.5)).score), 4)
            if "transformer" in by_name else None
        )

        evidence_summary["product"] = {
            "label": label,
            "confidence": round(confidence, 4),
            "confidence_band": confidence_band,
            "calibrated_probability": round(score, 4),
            "explanation": _build_explanation(
                label=label,
                score=score,
                l1=l1_val,
                l2=l2_val,
                l3=l3_val,
                word_count=word_count,
                short_text_min=SHORT_TEXT_WORD_MIN,
            ),
            "signals": {
                "l1_perplexity": l1_val,
                "l2_stylometry": l2_val,
                "l3_transformer": l3_val,
            },
            "meta_mode": "lr_calibrated" if used_lr_meta else "fixed_weight_fallback",
            "model_version": MODEL_VERSION,
        }

        return EnsembleResult(
            score=round(score, 4),
            label=label,
            confidence=confidence,
            layer_results=layer_results,
            feature_vector=feature_vector,
            evidence_summary=evidence_summary,
        )


# ── Evidence helpers ─────────────────────────────────────────


def _build_explanation(
    *,
    label: str,
    score: float,
    l1: float,
    l2: float,
    l3: float | None,
    word_count: int,
    short_text_min: int,
) -> str:
    """One-sentence human-readable explanation. Pure rule-based."""
    if label == "UNCERTAIN" and word_count < short_text_min:
        return (
            f"Text is too short ({word_count} words, minimum {short_text_min}) "
            f"for a reliable assessment."
        )
    if label == "UNCERTAIN":
        return (
            f"The detection signals are mixed (probability {score:.0%}). "
            f"No confident determination can be made."
        )

    reasons: list[str] = []
    if l3 is not None:
        if l3 >= 0.70:
            reasons.append("the language model detected AI-typical patterns")
        elif l3 <= 0.30:
            reasons.append("the language model found natural human writing patterns")

    if l1 >= 0.50:
        reasons.append("text predictability is high (low perplexity)")
    elif l1 <= 0.15:
        reasons.append("text has natural unpredictability")

    if l2 >= 0.45:
        reasons.append("writing style shows AI-typical characteristics")
    elif l2 <= 0.20:
        reasons.append("writing style appears natural")

    if not reasons:
        reasons.append(
            "multiple signals indicate AI-generated content"
            if label == "AI"
            else "multiple signals indicate human-written content"
        )

    if label == "AI":
        lead = f"Likely AI-generated ({score:.0%} probability)"
    else:
        lead = f"Likely human-written ({1 - score:.0%} probability)"
    return f"{lead}: {reasons[0]}."


def _build_top_signals(layer_results: list[LayerResult]) -> list[dict]:
    """Extract the most important evidence signals for UI display."""
    signals: list[dict] = []
    by_name = {r.layer_name: r for r in layer_results}

    l1 = by_name.get("perplexity")
    if l1 and not l1.error:
        ppl = l1.evidence.get("mean_perplexity", 0)
        if ppl < 50:
            signals.append({"signal": "Very low perplexity", "value": f"{ppl:.0f}", "weight": "high"})
        burst = l1.evidence.get("burstiness", 100)
        if burst < 20:
            signals.append({
                "signal": "Uniform sentence flow (low burstiness)",
                "value": f"{burst:.1f}",
                "weight": "medium",
            })

    l2 = by_name.get("stylometry")
    if l2 and not l2.error:
        hedge = l2.evidence.get("ai_hedge_rate", 0)
        if hedge > 0.01:
            signals.append({
                "signal": "High AI hedge word usage", "value": f"{hedge:.3f}", "weight": "medium",
            })
        em_dash = l2.evidence.get("em_dash_rate", 0)
        if em_dash > 0.005:
            signals.append({
                "signal": "Excessive em-dash usage", "value": f"{em_dash:.3f}", "weight": "low",
            })
        ttr = l2.evidence.get("type_token_ratio", 1.0)
        if ttr < 0.45:
            signals.append({
                "signal": "Narrow vocabulary (low TTR)", "value": f"{ttr:.3f}", "weight": "medium",
            })

    return signals[:8]


def _merge_sentence_scores(layer_results: list[LayerResult]) -> list[dict]:
    """Prefer L3 (transformer) sentence scores; fall back to L1 (perplexity)."""
    l3 = next((r for r in layer_results if r.layer_name == "transformer"), None)
    if not l3 or not l3.sentence_scores:
        l1 = next((r for r in layer_results if r.layer_name == "perplexity"), None)
        if l1 and l1.sentence_scores:
            return [
                {"text": s.text, "score": s.score, "evidence": s.evidence}
                for s in l1.sentence_scores
            ]
        return []
    return [
        {"text": s.text, "score": s.score, "evidence": s.evidence}
        for s in l3.sentence_scores
    ]


__all__ = [
    "TextDetector",
    "MODEL_VERSION",
    "ZONE_AI_THRESHOLD",
    "ZONE_HUMAN_THRESHOLD",
    "SHORT_TEXT_WORD_MIN",
]
