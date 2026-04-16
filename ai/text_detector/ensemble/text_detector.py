"""
Full text detection ensemble — combines all four layers + meta-classifier.

This is the single entry point for text detection at inference time.
The backend AI Detection Service calls TextDetector.analyze(text).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from ..layers.base import LayerResult
from ..layers.layer1_perplexity import PerplexityLayer
from ..layers.layer2_stylometry import StylometryLayer
from ..layers.layer3_semantic import SemanticLayer
from ..layers.layer4_adversarial import AdversarialLayer
from .meta_classifier import MetaClassifier, EnsembleResult, build_feature_vector

log = structlog.get_logger(__name__)

# Production model-version string. Bumped manually whenever the
# inference stack changes in a way that would affect score semantics
# (new weights, new calibrator, new layers, threshold change).
# Consumed by backend/app/observability/prediction_log.py so every
# logged prediction is traceable to a specific model configuration.
MODEL_VERSION = "3.1-reliability-gated"

# Score thresholds — adaptive based on number of active layers.
#
# 3-layer AI threshold was FIT on val data via grid search.
# Source: scripts/fit_ensemble_weights.py -> fit_weights.json
#   git_sha: fc64addaa53bfd47e98bfb14db7c6ebea887a00f
#   val_split: datasets/processed/val.parquet (n=2000)
#   val F1 at this threshold: 0.99692
#   test F1 verify (not used for selection): 0.99447
# DO NOT edit the 3-layer AI low-bound by hand — re-run the fit script
# and update this file and fit_weights.json in lockstep.
# UNCERTAIN band kept symmetric around the threshold at +/- 0.10.
# The 2-layer and 4-layer rows are unfit (2-layer is fallback-only;
# 4-layer is unreachable until an L4 checkpoint is trained).
_THRESHOLDS_BY_LAYERS = {
    2: {"AI": (0.55, 1.00), "UNCERTAIN": (0.30, 0.55), "HUMAN": (0.00, 0.30)},
    3: {"AI": (0.41, 1.00), "UNCERTAIN": (0.31, 0.41), "HUMAN": (0.00, 0.31)},
    4: {"AI": (0.75, 1.00), "UNCERTAIN": (0.40, 0.75), "HUMAN": (0.00, 0.40)},
}

# Default for meta-classifier (already calibrated to full range)
LABEL_THRESHOLDS = _THRESHOLDS_BY_LAYERS[4]


def _score_to_label(score: float, active_layers: int = 4) -> str:
    thresholds = _THRESHOLDS_BY_LAYERS.get(active_layers, LABEL_THRESHOLDS)
    for label, (low, high) in thresholds.items():
        if low <= score < high:
            return label
    return "UNCERTAIN"


def _score_to_confidence(score: float) -> float:
    """Convert probability to confidence (distance from 0.5)."""
    return round(abs(score - 0.5) * 2, 4)


class TextDetector:
    """
    Main text detection ensemble.

    load_models() is called once at worker startup (heavy).
    analyze() is called per-request (fast, ~200–500ms).
    """

    def __init__(
        self,
        transformer_checkpoint: Path | None = None,
        adversarial_checkpoint: Path | None = None,
        meta_checkpoint: Path | None = None,
        device: str | None = None,
        meta_lr_path: Path | None = None,
        meta_calibrator_path: Path | None = None,
    ) -> None:
        self._layer1 = PerplexityLayer(device=device)
        self._layer2 = StylometryLayer(use_spacy=True)
        self._transformer_checkpoint = transformer_checkpoint
        self._adversarial_checkpoint = adversarial_checkpoint
        self._layer3: SemanticLayer | None = None
        self._layer4: AdversarialLayer | None = None
        self._meta   = MetaClassifier()
        self._meta_checkpoint = meta_checkpoint
        self._loaded = False
        self._active_layers: list[int] = []  # indices of active layers
        self._device = device
        # Stage 2: learned LogisticRegression meta + isotonic calibration.
        # Both paths are optional. If both load successfully, analyze()
        # uses the calibrated probability as the ensemble score. If
        # either fails to load, the detector falls back to the Stage 1
        # fixed-weight combiner below — no breaking change.
        self._meta_lr_path = meta_lr_path
        self._meta_calibrator_path = meta_calibrator_path
        self._lr_meta = None          # sklearn LogisticRegression
        self._lr_calibrator = None    # sklearn CalibratedClassifierCV
        self._lr_threshold: float = 0.5
        self._lr_feature_order: list[str] = ["l1_score", "l2_score", "l3_score"]

    def load_models(self) -> None:
        """Load all model weights. Call once at startup."""
        log.info("loading_text_detector_models")

        # Pin seeds for reproducible inference across requests
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # L1 (perplexity) and L2 (stylometry) always load — no training needed
        self._layer1.load_model()
        self._layer2.load_model()
        self._active_layers = [0, 1]

        # L3 (semantic/transformer) — only load if fine-tuned checkpoint exists
        if self._transformer_checkpoint and self._transformer_checkpoint.exists():
            self._layer3 = SemanticLayer(
                checkpoint_path=self._transformer_checkpoint, device=self._device
            )
            self._layer3.load_model()
            self._active_layers.append(2)
            log.info("layer3_semantic_loaded", checkpoint=str(self._transformer_checkpoint))
        else:
            log.warning("layer3_skipped", reason="no fine-tuned checkpoint at "
                        + str(self._transformer_checkpoint or "None"))

        # L4 (adversarial) — only load if fine-tuned checkpoint exists
        if self._adversarial_checkpoint and self._adversarial_checkpoint.exists():
            self._layer4 = AdversarialLayer(
                checkpoint_path=self._adversarial_checkpoint, device=self._device
            )
            self._layer4.load_model()
            self._active_layers.append(3)
            log.info("layer4_adversarial_loaded")
        else:
            log.warning("layer4_skipped — no fine-tuned checkpoint, would produce random output")

        if self._meta_checkpoint and self._meta_checkpoint.exists():
            self._meta = MetaClassifier.load(self._meta_checkpoint)
        else:
            log.warning("meta_classifier_checkpoint_not_found — using heuristic fallback")

        # ── Stage 2: learned LR meta + isotonic calibration (optional) ──
        # Both the LR and the calibrator must load for the learned path
        # to activate. If either file is missing or fails to unpickle,
        # we fall back to the Stage 1 fixed-weight combiner — the
        # detector never breaks on a missing meta artifact.
        #
        # Auto-discovery: if the caller did not pass explicit meta paths
        # (e.g. Stage 1 evaluate_end_to_end.py, which predates Stage 2
        # and has no meta-related kwargs), try to find them next to the
        # transformer checkpoint. The Stage 2 training script writes
        # meta_classifier.joblib / meta_calibrator.joblib under
        # ai/text_detector/checkpoints/, which is the parent of
        # transformer_checkpoint.parent. This keeps evaluate_end_to_end.py
        # untouched while still letting the meta activate end-to-end.
        if (
            self._meta_lr_path is None
            and self._meta_calibrator_path is None
            and self._transformer_checkpoint is not None
        ):
            try:
                _meta_base = self._transformer_checkpoint.parent.parent
                _cand_lr = _meta_base / "meta_classifier.joblib"
                _cand_cal = _meta_base / "meta_calibrator.joblib"
                if _cand_lr.exists() and _cand_cal.exists():
                    self._meta_lr_path = _cand_lr
                    self._meta_calibrator_path = _cand_cal
                    log.info(
                        "lr_meta_auto_discovered",
                        lr_path=str(_cand_lr),
                        calibrator_path=str(_cand_cal),
                    )
            except Exception as exc:
                log.warning("lr_meta_auto_discovery_failed", error=str(exc))

        if (
            self._meta_lr_path is not None
            and self._meta_calibrator_path is not None
            and self._meta_lr_path.exists()
            and self._meta_calibrator_path.exists()
        ):
            try:
                import joblib
                lr_obj = joblib.load(self._meta_lr_path)
                cal_bundle = joblib.load(self._meta_calibrator_path)
                # cal_bundle is the dict written by train_meta_classifier.py:
                #   {"calibrator": CalibratedClassifierCV,
                #    "threshold": float, "feature_order": [...]}
                if isinstance(cal_bundle, dict) and "calibrator" in cal_bundle:
                    self._lr_meta = lr_obj
                    self._lr_calibrator = cal_bundle["calibrator"]
                    self._lr_threshold = float(cal_bundle.get("threshold", 0.5))
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
                else:
                    log.warning(
                        "lr_meta_calibrator_bundle_malformed — "
                        "expected dict with 'calibrator' key; falling back"
                    )
                    self._lr_meta = None
                    self._lr_calibrator = None
            except Exception as exc:
                log.error("lr_meta_load_failed", error=str(exc))
                self._lr_meta = None
                self._lr_calibrator = None
        else:
            log.info(
                "lr_meta_not_present",
                reason="meta_classifier.joblib / meta_calibrator.joblib not on disk; "
                       "using Stage 1 fixed-weight combiner",
            )

        active_count = len(self._active_layers)
        log.info("text_detector_ready", active_layers=active_count,
                 layers=self._active_layers)
        if active_count < 4:
            log.info("running_in_mvp_mode — L1+L2 only. Fine-tune L3/L4 for full accuracy.")

        self._loaded = True

    def analyze(self, text: str) -> EnsembleResult:
        """
        Run full 4-layer ensemble analysis on text.

        Returns EnsembleResult with:
          - score: calibrated [0, 1] AI probability
          - label: "AI" | "UNCERTAIN" | "HUMAN"
          - confidence: [0, 1]
          - layer_results: individual layer outputs
          - feature_vector: 26-dim vector
          - evidence_summary: key signals for the UI
        """
        if not self._loaded:
            raise RuntimeError("Call load_models() first.")

        # ── Run active layers (safe — errors return neutral 0.5) ──
        layer_results: list[LayerResult] = [
            self._layer1.analyze_safe(text),
            self._layer2.analyze_safe(text),
        ]
        # Only run L3/L4 if they have fine-tuned checkpoints loaded
        if self._layer3 is not None:
            layer_results.append(self._layer3.analyze_safe(text))
        if self._layer4 is not None:
            layer_results.append(self._layer4.analyze_safe(text))

        # ── Build feature vector ────────────────────────────────
        feature_vector = build_feature_vector(layer_results, text)

        # ── Score selection: learned meta > XGB meta > fixed weights ──
        # Order of precedence:
        #   1. Stage 2: LR + isotonic calibrator (if both loaded)
        #   2. Legacy: XGBoost MetaClassifier (if fitted — unused in prod)
        #   3. Stage 1: fixed-weight combiner (authoritative fallback)
        used_lr_meta = False
        if self._lr_meta is not None and self._lr_calibrator is not None:
            # Stage 2: learned LogisticRegression + isotonic calibration.
            # Input features are the per-layer raw scores in the order
            # frozen at training time (stored in _lr_feature_order).
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
            except Exception as exc:
                # If the learned path blows up at inference time, do
                # NOT crash the request — fall through to the fixed
                # weights below. This is the backward-compat guarantee.
                log.error("lr_meta_inference_failed_falling_back", error=str(exc))
                used_lr_meta = False

        if not used_lr_meta:
            if self._meta._is_fitted:
                score = self._meta.predict(feature_vector)
            else:
                # Stage 1 fallback: weighted average. Weights for the 3-layer
                # production path were FIT on val data via grid search.
                # Source: scripts/fit_ensemble_weights.py -> fit_weights.json
                #   git_sha: fc64addaa53bfd47e98bfb14db7c6ebea887a00f
                #   val F1 at these weights + threshold 0.41: 0.99692
                #   test F1 verify (not used for selection): 0.99447
                # DO NOT edit the 3-layer weights by hand — re-run the fit
                # script and update this file and fit_weights.json in lockstep.
                # The 2-layer and 4-layer rows are unfit.
                active_count = len(layer_results)
                if active_count == 2:
                    weights = [0.50, 0.50]
                elif active_count == 3:
                    weights = [0.20, 0.35, 0.45]
                else:
                    weights = [0.20, 0.20, 0.35, 0.25]
                scores = [r.score for r in layer_results]
                score = sum(s * w for s, w in zip(scores, weights))
                score = max(0.01, min(0.99, score))

        active_count = len(layer_results)

        # ── Reliability-gated 3-zone decision policy ────────────
        #
        # Optimised for per-request reliability (precision of returned
        # labels) over raw coverage. Three zones:
        #
        #   score >= 0.70  →  AI       (high confidence)
        #   score <= 0.30  →  HUMAN    (high confidence)
        #   else           →  UNCERTAIN (abstain — not reliable enough)
        #
        # Two additional GATING RULES can push an otherwise-decisive
        # prediction into UNCERTAIN:
        #
        #   G1  Short text (<50 words): L1 perplexity and L2 stylometry
        #       have no statistical power on short inputs. Flag as
        #       UNCERTAIN rather than risking a noisy prediction.
        #
        #   G2  Layer disagreement: if L2 (stylometry) and L3
        #       (transformer) disagree by more than 0.40 on the raw
        #       score (one says "AI", the other says "human"), the
        #       ensemble is internally contradicted. Flag UNCERTAIN.
        #
        # These thresholds are intentionally conservative. The trade
        # is: lower coverage (more UNCERTAINs) in exchange for higher
        # reliability on the predictions we DO return.

        ZONE_AI_THRESHOLD = 0.70
        ZONE_HUMAN_THRESHOLD = 0.30
        SHORT_TEXT_WORD_MIN = 50
        DISAGREEMENT_THRESHOLD = 0.40

        # Default zone from the score
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

        # G2: L2–L3 disagreement gate
        by_name_for_gate = {r.layer_name: r for r in layer_results}
        l2_r = by_name_for_gate.get("stylometry")
        l3_r = by_name_for_gate.get("transformer")
        if (
            l2_r is not None and not l2_r.error
            and l3_r is not None and not l3_r.error
        ):
            l2_score = float(l2_r.score)
            l3_score = float(l3_r.score)
            if abs(l2_score - l3_score) > DISAGREEMENT_THRESHOLD and label != "UNCERTAIN":
                label = "UNCERTAIN"

        confidence = _score_to_confidence(score)

        # ── Evidence summary for the UI ─────────────────────────
        by_name = {r.layer_name: r for r in layer_results}
        evidence_summary: dict[str, Any] = {
            "layer_scores": {
                "perplexity":   by_name.get("perplexity", LayerResult("perplexity", 0.5)).score,
                "stylometry":   by_name.get("stylometry", LayerResult("stylometry", 0.5)).score,
                "transformer":  by_name.get("transformer", LayerResult("transformer", 0.5)).score if "transformer" in by_name else None,
                "adversarial":  by_name.get("adversarial", LayerResult("adversarial", 0.5)).score if "adversarial" in by_name else None,
            },
            "layer_errors": {
                r.layer_name: r.error
                for r in layer_results if r.error
            },
            "top_signals": _build_top_signals(layer_results),
            "sentence_scores": _merge_sentence_scores(layer_results),
        }

        return EnsembleResult(
            score=round(score, 4),
            label=label,
            confidence=confidence,
            layer_results=layer_results,
            feature_vector=feature_vector,
            evidence_summary=evidence_summary,
        )


def _build_top_signals(layer_results: list[LayerResult]) -> list[dict]:
    """Extract the most important evidence signals for UI display."""
    signals = []
    by_name = {r.layer_name: r for r in layer_results}

    l1 = by_name.get("perplexity")
    if l1 and not l1.error:
        ppl = l1.evidence.get("mean_perplexity", 0)
        if ppl < 50:
            signals.append({"signal": "Very low perplexity", "value": f"{ppl:.0f}", "weight": "high"})
        burst = l1.evidence.get("burstiness", 100)
        if burst < 20:
            signals.append({"signal": "Uniform sentence flow (low burstiness)", "value": f"{burst:.1f}", "weight": "medium"})

    l2 = by_name.get("stylometry")
    if l2 and not l2.error:
        hedge = l2.evidence.get("ai_hedge_rate", 0)
        if hedge > 0.01:
            signals.append({"signal": "High AI hedge word usage", "value": f"{hedge:.3f}", "weight": "medium"})
        em_dash = l2.evidence.get("em_dash_rate", 0)
        if em_dash > 0.005:
            signals.append({"signal": "Excessive em-dash usage", "value": f"{em_dash:.3f}", "weight": "low"})
        ttr = l2.evidence.get("type_token_ratio", 1.0)
        if ttr < 0.45:
            signals.append({"signal": "Narrow vocabulary (low TTR)", "value": f"{ttr:.3f}", "weight": "medium"})

    return signals[:8]   # top 8 signals for the UI


def _merge_sentence_scores(layer_results: list[LayerResult]) -> list[dict]:
    """
    Merge per-sentence scores from all layers into a unified list.
    For each sentence, average available layer scores.
    """
    # Use Layer 3 (transformer) as the primary sentence scorer
    l3 = next((r for r in layer_results if r.layer_name == "transformer"), None)
    if not l3 or not l3.sentence_scores:
        # Fall back to Layer 1
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
