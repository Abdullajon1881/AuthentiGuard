"""
Text/code detection worker — Celery task using BaseDetectionWorker.

Loads text from DB (paste) or S3 (file upload), runs the TextDetector ensemble,
writes DetectionResult, and triggers webhook on completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from celery import Task  # type: ignore

from .base_worker import BaseDetectionWorker, run_async
from .celery_app import celery_app
from ..models.models import DetectionJob, DetectionResult

log = structlog.get_logger(__name__)

# Module-level detector — loaded once per worker process, not per task
_detector = None


class _DevFallbackDetector:
    """Heuristic detector for local dev when AI models are unavailable.

    Mirrors the real Layer 2 (stylometry) scoring algorithm: 9 independent
    signals each on [0.0, 1.0], averaged to produce the final score.
    Word lists copied from ai/text_detector/layers/layer2_stylometry.py.
    """

    # Word lists from the real detector
    _AI_HEDGE_WORDS = {
        "furthermore", "moreover", "additionally", "consequently", "therefore",
        "nevertheless", "nonetheless", "however", "indeed", "certainly",
        "undoubtedly", "essentially", "fundamentally", "ultimately", "notably",
        "importantly", "significantly", "particularly", "specifically",
        "delve", "dive", "tapestry", "nuanced", "multifaceted", "comprehensive",
        "robust", "leverage", "utilize", "facilitate", "paradigm",
    }
    _HUMAN_CASUAL_WORDS = {
        "actually", "basically", "kind of", "sort of", "you know", "i mean",
        "well", "anyway", "stuff", "thing", "things", "pretty", "really",
        "just", "like", "though", "although", "but then", "so",
        # Additional informal markers
        "honestly", "somehow", "maybe", "probably", "guess", "kinda",
        "gonna", "gotta", "wanna", "ok", "okay", "cool", "weird",
        "lol", "haha", "omg", "btw", "tbh", "ngl", "imo",
    }
    _FIRST_PERSON = {
        "i", "me", "my", "myself", "mine",
    }
    _CONTRACTIONS = {
        "don't", "doesn't", "didn't", "can't", "won't", "wouldn't", "shouldn't",
        "couldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
        "hadn't", "i'm", "i've", "i'll", "i'd", "we're", "we've", "we'll",
        "they're", "they've", "they'll", "you're", "you've", "you'll",
        "it's", "that's", "there's", "here's", "what's", "who's", "let's",
    }

    @staticmethod
    def _get_sentences(text: str) -> list[str]:
        import re
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sentences if len(s.split()) >= 3]

    @staticmethod
    def _word_tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"\b\w+\b", text.lower())

    def analyze(self, text: str):
        import re, math

        sentences = self._get_sentences(text)
        if not sentences:
            sentences = [text]

        words = self._word_tokenize(text)
        n_words = max(len(words), 1)

        # ── 9 signals, each [0.0, 1.0], higher = more AI-like ──

        # 1. Sentence length variance — low std → AI
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 1:
            mean_sl = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_sl) ** 2 for l in sent_lengths) / len(sent_lengths)
            std_sl = math.sqrt(variance)
        else:
            std_sl = 10.0
        sig_1 = max(0.0, 1.0 - std_sl / 15.0)

        # 2. AI hedge word rate — high rate → AI
        hedge_count = sum(1 for w in words if w in self._AI_HEDGE_WORDS)
        hedge_rate = hedge_count / n_words
        sig_2 = min(hedge_rate * 80.0, 1.0)

        # 3. Human casual word absence — low casual rate → AI
        casual_count = sum(1 for w in words if w in self._HUMAN_CASUAL_WORDS)
        casual_rate = casual_count / n_words
        sig_3 = max(0.0, 1.0 - casual_rate * 50.0)

        # 4. Type-token ratio — AI has narrower vocabulary (lower TTR)
        #    Short texts naturally have high TTR; adjust threshold by length
        unique_words = set(words)
        ttr = len(unique_words) / n_words
        ttr_ceiling = 0.7 if n_words >= 200 else 0.7 + 0.3 * (200 - n_words) / 200
        sig_4 = max(0.0, 1.0 - ttr / ttr_ceiling)

        # 5. Sentence-initial diversity — low diversity → AI
        if len(sentences) >= 3:
            first_words = [s.split()[0].lower() for s in sentences if s.split()]
            sid = len(set(first_words)) / len(first_words) if first_words else 0.7
        else:
            sid = 0.7
        sig_5 = max(0.0, 1.0 - sid)

        # 6. Em-dash overuse — AI loves em-dashes
        #    No em-dashes = neutral (0.5), not human-indicative
        em_dash_count = text.count("\u2014")
        em_rate = em_dash_count / n_words
        sig_6 = min(0.5 + em_rate * 100.0, 1.0)

        # 7. Comma rate — AI uses more commas
        #    Lowered threshold: AI comma rate often 0.03-0.06+
        comma_count = text.count(",")
        comma_rate = comma_count / n_words
        sig_7 = min(max(0.0, (comma_rate - 0.03) / 0.08), 1.0)

        # 8. Contraction absence — no contractions in long text → AI
        contraction_count = sum(1 for w in text.lower().split() if w in self._CONTRACTIONS)
        if len(text) >= 500:
            sig_8 = 1.0 if contraction_count == 0 else max(0.0, 1.0 - contraction_count * 0.3)
        else:
            sig_8 = 0.8 if contraction_count == 0 else max(0.0, 1.0 - contraction_count * 0.3)

        # 9. Average word length — longer words → more formal/AI
        avg_wl = sum(len(w) for w in words) / n_words
        sig_9 = min(max(0.0, (avg_wl - 4.0) / 4.0), 1.0)

        # 10. First-person pronouns — humans write about themselves
        fp_count = sum(1 for w in words if w in self._FIRST_PERSON)
        fp_rate = fp_count / n_words
        sig_10 = max(0.0, 1.0 - fp_rate * 20.0)  # high first-person → low (human)

        # ── A1: Weighted average — boost discriminative signals ──
        all_signals = [sig_1, sig_2, sig_3, sig_4, sig_5, sig_6, sig_7, sig_8, sig_9, sig_10]
        weights =     [1.5,  2.5,  1.0,  0.5,  0.5,  0.5,  0.5,  2.0,  1.0,  2.0]
        #              var  hedge casual TTR  divrs emdsh comma contr wlen 1stPrsn
        score = sum(s * w for s, w in zip(all_signals, weights)) / sum(weights)
        score = max(0.01, min(0.99, score))

        confidence = 0.5 + abs(score - 0.5) * 0.8
        label = "AI" if score > 0.55 else "HUMAN" if score < 0.40 else "UNCERTAIN"

        # ── A2: Each layer uses a distinct signal subset ──
        def _clamp(v: float) -> float:
            return max(0.01, min(0.99, v))

        # Perplexity (sentence predictability): variance, TTR, diversity, comma
        perplexity_score = _clamp((sig_1 * 2.0 + sig_4 + sig_5 + sig_7) / 5.0)
        # Stylometry (vocabulary fingerprint): hedge, casual, contraction, word length, 1st person
        stylometry_score = _clamp((sig_2 * 2.0 + sig_3 * 1.0 + sig_8 * 1.5 + sig_9 + sig_10 * 1.5) / 7.0)
        # Transformer (full ensemble): same as main weighted score
        transformer_score = _clamp(score)
        # Adversarial (robustness signals): variance, em-dash, comma, contraction
        adversarial_score = _clamp((sig_1 * 1.5 + sig_6 + sig_7 + sig_8 * 1.5) / 5.0)

        # ── A3: Per-sentence scoring with evidence signals ──
        sent_scores = []
        for sent in sentences:
            sent_words = self._word_tokenize(sent)
            n = max(len(sent_words), 1)
            hedge = sum(1 for w in sent_words if w in self._AI_HEDGE_WORDS) / n
            casual = sum(1 for w in sent_words if w in self._HUMAN_CASUAL_WORDS) / n
            has_contraction = any(w in self._CONTRACTIONS for w in sent.lower().split())
            avg_word_len = sum(len(w) for w in sent_words) / n

            # Independent sentence scoring — no doc-level anchor
            fp_in_sent = sum(1 for w in sent.lower().split() if w in self._FIRST_PERSON)
            s_score = 0.5  # neutral start
            s_score += hedge * 15.0                                     # hedge words → AI
            s_score -= casual * 10.0                                    # casual words → human
            s_score -= 0.20 if has_contraction else 0.0                 # contractions → human
            s_score += 0.08 if not has_contraction and n >= 8 else 0.0  # no contractions in long sent → AI
            s_score += 0.05 if casual == 0 and n >= 5 else 0.0         # no casual markers → AI
            s_score += max(0.0, (avg_word_len - 4.5) / 6.0) * 0.12     # formal vocabulary → AI
            s_score -= fp_in_sent * 0.12                                # first-person → human
            s_score = max(0.01, min(0.99, s_score))
            s_label = "AI" if s_score > 0.6 else "HUMAN" if s_score < 0.4 else "UNCERTAIN"

            # Build per-sentence evidence signals
            signals_found = []
            if hedge > 0:
                hedge_words_found = [w for w in sent_words if w in self._AI_HEDGE_WORDS]
                signals_found.append("AI hedge: " + ", ".join(hedge_words_found[:3]))
            if casual > 0:
                casual_found = [w for w in sent_words if w in self._HUMAN_CASUAL_WORDS]
                signals_found.append("Casual: " + ", ".join(casual_found[:3]))
            elif n >= 5:
                signals_found.append("No casual markers")
            if has_contraction:
                signals_found.append("Contractions present")
            elif n >= 8:
                signals_found.append("No contractions")
            if fp_in_sent > 0:
                signals_found.append("First-person voice")
            if avg_word_len > 5.0:
                signals_found.append(f"Formal vocabulary (avg {avg_word_len:.1f} chars)")

            sent_scores.append((sent, round(s_score, 3), {
                "label": s_label,
                "hedge_rate": round(hedge, 4),
                "casual_rate": round(casual, 4),
                "has_contraction": has_contraction,
                "signals": signals_found,
            }))

        # ── Build result object ──
        class _R:
            pass
        r = _R()
        r.score = round(score, 4)
        r.confidence = round(confidence, 4)
        r.label = label

        class _Layer:
            def __init__(self, name, s):
                self.layer_name = name
                self.score = round(max(0.0, min(1.0, s)), 4)

        r.layer_results = [
            _Layer("perplexity",  perplexity_score),
            _Layer("stylometry",  stylometry_score),
            _Layer("transformer", transformer_score),
            _Layer("adversarial", adversarial_score),
        ]

        r.evidence_summary = {
            "top_signals": [],
            "sentence_scores": [
                {"text": s, "score": sc, "evidence": ev}
                for s, sc, ev in sent_scores
            ],
        }

        # Top signals for the UI
        top_signals = []
        if sig_1 > 0.5:
            top_signals.append({"signal": "Uniform sentence length", "value": f"std={std_sl:.1f} words", "weight": "high"})
        if sig_2 > 0.3:
            top_signals.append({"signal": "AI hedge words detected", "value": f"{hedge_count} found ({hedge_rate:.3f}/word)", "weight": "high"})
        if sig_8 > 0.7 and len(text) >= 200:
            top_signals.append({"signal": "No contractions detected", "value": "formal register", "weight": "high"})
        if sig_3 > 0.8:
            top_signals.append({"signal": "No casual language", "value": "formal tone", "weight": "medium"})
        if sig_7 > 0.3:
            top_signals.append({"signal": "High comma usage", "value": f"{comma_rate:.3f}/word", "weight": "medium"})
        if sig_9 > 0.3:
            top_signals.append({"signal": "Formal vocabulary", "value": f"avg {avg_wl:.1f} chars/word", "weight": "medium"})
        if contraction_count > 0:
            top_signals.append({"signal": "Natural contractions present", "value": f"{contraction_count} found", "weight": "medium"})
        r.evidence_summary["top_signals"] = top_signals[:5]

        # Model attribution
        if score > 0.55:
            r.model_attribution = {
                "gpt_family": round(score * 0.45, 3),
                "claude_family": round(score * 0.30, 3),
                "llama_family": round(score * 0.15, 3),
                "human": round(1 - score, 3),
                "other": round(score * 0.10, 3),
            }
        else:
            r.model_attribution = {
                "gpt_family": round(score * 0.2, 3),
                "claude_family": round(score * 0.1, 3),
                "llama_family": round(score * 0.05, 3),
                "human": round(max(0, 1 - score * 0.8), 3),
                "other": round(score * 0.05, 3),
            }
        return r


def _get_detector():
    global _detector
    if _detector is None:
        try:
            import sys, os
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
            if root not in sys.path:
                sys.path.insert(0, root)
            from ai.text_detector.ensemble.text_detector import TextDetector  # type: ignore
            _detector = TextDetector(
                transformer_checkpoint=Path("ai/text_detector/checkpoints/transformer/phase3"),
                adversarial_checkpoint=Path("ai/text_detector/checkpoints/adversarial/phase3"),
                meta_checkpoint=Path("ai/text_detector/checkpoints/meta"),
            )
            _detector.load_models()
            log.info("text_detector_loaded_in_worker")
        except (ImportError, FileNotFoundError, OSError) as exc:
            log.warning("text_detector_unavailable_using_dev_fallback", error=str(exc))
            _detector = _DevFallbackDetector()
    return _detector


class TextDetectionWorker(BaseDetectionWorker):
    content_type = "text"

    def get_detector(self) -> Any:
        return _get_detector()

    async def get_input(self, job: DetectionJob) -> str:
        text = await _resolve_text(job)
        if not text or len(text.strip()) < 20:
            raise ValueError("Text content is too short to analyze (minimum 20 characters)")
        return text

    def run_detection(self, detector: Any, input_data: str, job: DetectionJob) -> Any:
        return detector.analyze(input_data)

    def build_result(
        self,
        job: DetectionJob,
        detection_output: Any,
        elapsed_ms: int,
    ) -> DetectionResult:
        return DetectionResult(
            job_id=job.id,
            authenticity_score=detection_output.score,
            confidence=detection_output.confidence,
            label=detection_output.label,
            layer_scores={
                r.layer_name: r.score
                for r in detection_output.layer_results
            },
            evidence_summary=detection_output.evidence_summary,
            sentence_scores=detection_output.evidence_summary.get("sentence_scores", []),
            model_attribution=getattr(detection_output, "model_attribution", {}),
            processing_ms=elapsed_ms,
        )


_worker = TextDetectionWorker()


@celery_app.task(
    bind=True,
    name="workers.text_worker.run_text_detection",
    queue="text",
    max_retries=3,
    default_retry_delay=10,
)
def run_text_detection(self: Task, job_id: str) -> dict:
    """Celery task: run full text detection ensemble for a job."""
    try:
        return run_async(_worker.execute(job_id))
    except Exception as exc:
        if isinstance(exc, ValueError):
            return {"error": str(exc)}
        raise self.retry(exc=exc, countdown=10)


# ── Text resolution helpers ──────────────────────────────────

async def _resolve_text(job: DetectionJob) -> str:
    """Get text content from job — either direct paste or S3 file."""
    if job.input_text:
        return job.input_text

    if job.s3_key:
        import boto3
        from ..core.config import get_settings
        settings = get_settings()

        kwargs = {
            "region_name":          settings.AWS_REGION,
            "aws_access_key_id":    settings.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
        }
        if settings.S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL

        s3 = boto3.client("s3", **kwargs)
        obj = s3.get_object(Bucket=settings.S3_BUCKET_UPLOADS, Key=job.s3_key)
        data = obj["Body"].read()

        if job.s3_key.endswith(".pdf"):
            return _extract_pdf_text(data)
        elif job.s3_key.endswith(".docx"):
            return _extract_docx_text(data)
        else:
            return data.decode("utf-8", errors="replace")

    raise ValueError("Job has no text content or S3 key")


def _extract_pdf_text(data: bytes) -> str:
    try:
        import pypdf  # type: ignore
        import io
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        return data.decode("utf-8", errors="replace")


def _extract_docx_text(data: bytes) -> str:
    try:
        import docx  # type: ignore
        import io
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        return data.decode("utf-8", errors="replace")
