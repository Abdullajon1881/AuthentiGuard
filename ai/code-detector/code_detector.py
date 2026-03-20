"""
Step 78: Full code AI detection pipeline.
Target: 85–90% accuracy on AI vs human code classification.

Pipeline:
  source code → detect language → AST analysis → naming analysis →
  comment analysis → code transformer → combine → calibrate → CodeDetectionResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from .pipeline.ast_analyzer import (
    analyse_ast, ASTFeatures, SUPPORTED_LANGUAGES, _empty_ast_features,
)
from .features.naming_analyzer import analyse_naming_patterns, NamingFeatures
from .features.comment_analyzer import analyse_comment_style, CommentFeatures
from .models.transformer import CodeTransformerClassifier

log = structlog.get_logger(__name__)


@dataclass
class CodeDetectionResult:
    """Final detection result for one code file or snippet."""
    score:         float
    label:         str
    confidence:    float
    language:      str
    ast_features:  ASTFeatures
    naming:        NamingFeatures
    comments:      CommentFeatures
    transformer_score: float
    evidence:      dict[str, Any]
    processing_ms: int


class CodeDetector:
    """
    Full code AI detection pipeline.
    Combines AST analysis, naming patterns, comment style, and
    a fine-tuned CodeBERT transformer.
    """

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device: str | None = None,
        use_transformer: bool = True,
    ) -> None:
        self._transformer    = CodeTransformerClassifier(checkpoint_path=checkpoint_path, device=device)
        self._use_transformer = use_transformer
        self._loaded          = False

    def load_models(self) -> None:
        if self._use_transformer:
            try:
                self._transformer.load()
                log.info("code_transformer_loaded")
            except Exception as exc:
                log.warning("code_transformer_load_failed", error=str(exc))
                self._use_transformer = False
        self._loaded = True
        log.info("code_detector_ready")

    def analyze(self, source: str, filename: str = "code.py") -> CodeDetectionResult:
        if not self._loaded:
            raise RuntimeError("Call load_models() first")

        t_start = int(time.time() * 1000)

        # Detect language from extension
        ext      = Path(filename).suffix.lower()
        language = SUPPORTED_LANGUAGES.get(ext, "python")

        # Steps 73–75: Feature extraction
        ast_features = analyse_ast(source, language)
        naming       = analyse_naming_patterns(source, language)
        comments     = analyse_comment_style(
            source, language, n_functions=ast_features.n_functions
        )

        # Step 76: Transformer score
        if self._use_transformer:
            try:
                transformer_score = self._transformer.predict(source)
            except Exception as exc:
                log.warning("transformer_predict_failed", error=str(exc))
                transformer_score = 0.5
        else:
            transformer_score = self._heuristic_score(ast_features, naming, comments)

        # Combine all signals
        final_score = self._combine_scores(
            ast_features, naming, comments, transformer_score
        )

        label = (
            "AI"        if final_score >= 0.75 else
            "HUMAN"     if final_score <= 0.40 else
            "UNCERTAIN"
        )
        confidence    = round(abs(final_score - 0.5) * 2, 4)
        processing_ms = int(time.time() * 1000) - t_start

        evidence = self._build_evidence(
            ast_features, naming, comments, transformer_score, final_score
        )

        log.info("code_detection_complete",
                 filename=filename, language=language,
                 score=round(final_score, 4), label=label, ms=processing_ms)

        return CodeDetectionResult(
            score=round(final_score, 4),
            label=label,
            confidence=confidence,
            language=language,
            ast_features=ast_features,
            naming=naming,
            comments=comments,
            transformer_score=round(transformer_score, 4),
            evidence=evidence,
            processing_ms=processing_ms,
        )

    # ── Score combination ──────────────────────────────────────

    @staticmethod
    def _heuristic_score(
        ast: ASTFeatures,
        naming: NamingFeatures,
        comments: CommentFeatures,
    ) -> float:
        """
        Heuristic fallback when transformer is unavailable.
        Weights based on signal reliability from literature.
        """
        signals: list[float] = []

        # Naming patterns
        signals.append(naming.ai_pattern_score)

        # Comment style
        signals.append(comments.ai_comment_score)

        # Docstring coverage: >85% → AI
        if ast.n_functions > 2:
            doc_cov_signal = min(ast.docstring_coverage / 0.85, 1.0)
            signals.append(doc_cov_signal)

        # Low function length CV → AI (uniform function sizes)
        if ast.fn_length_cv > 0:
            cv_signal = max(0.0, 1.0 - ast.fn_length_cv / 0.5)
            signals.append(cv_signal)

        # Absence of human artifacts → AI
        artifact_count = ast.n_todos + ast.n_fixmes + ast.n_commented_code
        artifact_signal = max(0.0, 1.0 - artifact_count * 0.1)
        signals.append(artifact_signal)

        return float(np.mean(signals)) if signals else 0.5

    @staticmethod
    def _combine_scores(
        ast:          ASTFeatures,
        naming:       NamingFeatures,
        comments:     CommentFeatures,
        transformer:  float,
    ) -> float:
        """
        Combine transformer and heuristic signals.
        Transformer gets higher weight when available (trained end-to-end).
        Heuristic signals provide interpretability and robustness.
        """
        heuristic = CodeDetector._heuristic_score(ast, naming, comments)

        # Blend: 60% transformer, 40% heuristics
        combined = 0.60 * transformer + 0.40 * heuristic
        return float(np.clip(combined, 0.01, 0.99))

    @staticmethod
    def _build_evidence(
        ast:          ASTFeatures,
        naming:       NamingFeatures,
        comments:     CommentFeatures,
        transformer:  float,
        final:        float,
    ) -> dict[str, Any]:
        signals: list[dict] = []

        if naming.ai_pattern_score > 0.70:
            signals.append({"signal": "AI naming patterns detected",
                             "value": f"{naming.ai_pattern_score:.2f}",
                             "weight": "high"})
        if ast.docstring_coverage > 0.85 and ast.n_functions > 2:
            signals.append({"signal": "Unusually high docstring coverage",
                             "value": f"{ast.docstring_coverage:.0%}",
                             "weight": "medium"})
        if comments.ai_comment_score > 0.70:
            signals.append({"signal": "AI comment style detected",
                             "value": f"{comments.ai_comment_score:.2f}",
                             "weight": "medium"})
        if ast.n_todos == 0 and ast.n_fixmes == 0 and ast.n_commented_code == 0 \
                and ast.n_lines > 50:
            signals.append({"signal": "No human development artifacts",
                             "value": "none found",
                             "weight": "medium"})
        if ast.fn_length_cv < 0.35 and ast.n_functions >= 3:
            signals.append({"signal": "Suspiciously uniform function lengths",
                             "value": f"CV={ast.fn_length_cv:.2f}",
                             "weight": "low"})

        return {
            # AST
            "language":            ast.language,
            "n_lines":             ast.n_lines,
            "n_functions":         ast.n_functions,
            "n_classes":           ast.n_classes,
            "fn_length_mean":      ast.fn_length_mean,
            "fn_length_cv":        ast.fn_length_cv,
            "max_nesting_depth":   ast.max_nesting_depth,
            "docstring_coverage":  ast.docstring_coverage,
            "n_todos":             ast.n_todos,
            "n_fixmes":            ast.n_fixmes,
            "n_magic_numbers":     ast.n_magic_numbers,
            "n_commented_code":    ast.n_commented_code,

            # Naming
            "mean_name_length":        naming.mean_name_length,
            "abbreviation_rate":       naming.abbreviation_rate,
            "convention_consistency":  naming.convention_consistency,
            "single_letter_rate":      naming.single_letter_rate,
            "numeric_suffix_rate":     naming.numeric_suffix_rate,
            "naming_ai_score":         naming.ai_pattern_score,
            "sample_names":            naming.sample_names[:15],

            # Comments
            "capitalisation_rate":     comments.capitalisation_rate,
            "ends_with_period_rate":   comments.ends_with_period_rate,
            "google_style_ratio":      comments.google_style_ratio,
            "format_consistency":      comments.format_consistency,
            "informal_marker_count":   comments.informal_marker_count,
            "comment_ai_score":        comments.ai_comment_score,

            # Model
            "transformer_score":  transformer,
            "top_signals":        signals,
        }
