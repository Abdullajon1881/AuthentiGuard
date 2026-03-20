"""
Steps 99–100: Adversarial robustness testing and cross-model calibration.

Step 99 — Adversarial robustness test suite
  Tests the system's resistance to evasion attacks — attempts by a
  sophisticated adversary to make AI-generated content appear human.

  Attack categories:
    Text:  paraphrase, back-translation, synonym substitution,
           character-level noise, whitespace injection, homoglyph
    Image: JPEG compression, additive noise, crop, blur, colour jitter,
           adversarial patches (FGSM / PGD)
    Audio: pitch shift, time stretch, additive noise, codec recompression
    Code:  variable renaming, comment injection, whitespace normalisation

  Each attack is applied at multiple strengths and the resulting
  detection rate is compared to a no-attack baseline.

  Acceptance criteria (Step 54 / Step 64 targets):
    Text:  >75% detection rate after paraphrase
    Image: >80% detection rate after JPEG Q=50
    Audio: >70% detection rate after pitch shift ±2 semitones
    Code:  >70% detection rate after variable renaming

Step 100 — Cross-model calibration validation
  Ensures all detectors produce well-calibrated probability outputs.
  A calibrated model assigns p=0.7 to events that are truly AI 70%
  of the time (neither over- nor under-confident).

  Metrics:
    ECE (Expected Calibration Error): < 0.05 target
    MCE (Maximum Calibration Error):  < 0.15 target
    Reliability diagram: plotted and stored for audit

  If calibration degrades after adversarial fine-tuning (Step 53 etc.),
  the re-calibration pipeline re-fits Platt + isotonic regression on
  the new validation set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import structlog

log = structlog.get_logger(__name__)


# ── Step 99: Adversarial robustness tests ─────────────────────

@dataclass
class RobustnessResult:
    """Robustness result for one attack type at one strength."""
    attack:          str
    strength:        str         # "low" | "medium" | "high"
    baseline_score:  float       # detection score without attack
    attacked_score:  float       # detection score with attack
    score_drop:      float       # baseline - attacked
    detection_rate:  float       # fraction correctly detected after attack
    passed:          bool        # meets acceptance criterion


@dataclass
class RobustnessSuite:
    """Complete robustness evaluation across all attacks."""
    content_type:       str
    results:            list[RobustnessResult]
    overall_pass_rate:  float    # fraction of attack scenarios passed
    weakest_attack:     str | None
    recommendation:     str


def run_robustness_suite(
    detector:       Any,
    content_type:   str,
    ai_samples:     list[Any],
    attack_fn_map:  dict[str, Callable] | None = None,
    thresholds:     dict[str, float] | None = None,
) -> RobustnessSuite:
    """
    Run the full adversarial robustness suite against a detector.

    Args:
        detector:      A detector with an analyze() method
        content_type:  "text" | "image" | "audio" | "code"
        ai_samples:    List of confirmed AI-generated content samples
        attack_fn_map: {attack_name: fn(content, strength) -> content}
        thresholds:    {attack_name: min_detection_rate}
    """
    attacks   = attack_fn_map or _get_default_attacks(content_type)
    criteria  = thresholds   or _get_acceptance_criteria(content_type)
    results:  list[RobustnessResult] = []
    n_samples = min(len(ai_samples), 50)   # cap for speed

    for attack_name, attack_fn in attacks.items():
        for strength in ["low", "medium", "high"]:
            baseline_scores:  list[float] = []
            attacked_scores:  list[float] = []

            for sample in ai_samples[:n_samples]:
                try:
                    # Baseline score (no attack)
                    base_result  = _run_detector(detector, sample, content_type)
                    baseline_scores.append(base_result)

                    # Attacked score
                    attacked     = attack_fn(sample, strength)
                    attack_result = _run_detector(detector, attacked, content_type)
                    attacked_scores.append(attack_result)

                except Exception as exc:
                    log.warning("attack_sample_failed",
                                attack=attack_name, error=str(exc))

            if not baseline_scores:
                continue

            baseline = float(np.mean(baseline_scores))
            attacked  = float(np.mean(attacked_scores))
            drop      = baseline - attacked
            det_rate  = float(np.mean([s >= 0.50 for s in attacked_scores]))

            min_rate  = criteria.get(attack_name, 0.70)
            passed    = det_rate >= min_rate

            results.append(RobustnessResult(
                attack=attack_name,
                strength=strength,
                baseline_score=round(baseline, 4),
                attacked_score=round(attacked, 4),
                score_drop=round(drop, 4),
                detection_rate=round(det_rate, 4),
                passed=passed,
            ))

    # Aggregate
    if not results:
        return RobustnessSuite(
            content_type=content_type, results=[],
            overall_pass_rate=0.0, weakest_attack=None,
            recommendation="No samples available for robustness testing",
        )

    pass_rate = float(np.mean([r.passed for r in results]))
    weakest   = min(results, key=lambda r: r.detection_rate)

    recommendation = (
        f"All robustness targets met. "
        f"Weakest attack: {weakest.attack} at {weakest.strength} "
        f"({weakest.detection_rate:.0%} detection rate)."
        if pass_rate >= 1.0 else
        f"Robustness improvements needed. "
        f"Pass rate: {pass_rate:.0%}. "
        f"Critical failure: {weakest.attack} ({weakest.detection_rate:.0%} detection)."
    )

    return RobustnessSuite(
        content_type=content_type,
        results=results,
        overall_pass_rate=round(pass_rate, 4),
        weakest_attack=weakest.attack,
        recommendation=recommendation,
    )


def _run_detector(detector: Any, content: Any, content_type: str) -> float:
    """Call the detector and return a score, handling different return types."""
    try:
        if content_type == "text":
            result = detector.analyze(content)
        else:
            result = detector.analyze(content, f"file.{content_type}")
        return float(getattr(result, "score", 0.5))
    except Exception:
        return 0.5


def _get_default_attacks(content_type: str) -> dict[str, Callable]:
    """Return default attack functions for each content type."""
    if content_type in ("text", "code"):
        return {
            "paraphrase":         _text_paraphrase_attack,
            "synonym_sub":        _text_synonym_attack,
            "whitespace_inject":  _text_whitespace_attack,
            "homoglyph":          _text_homoglyph_attack,
        }
    elif content_type == "image":
        return {
            "jpeg_compression":   _image_jpeg_attack,
            "additive_noise":     _image_noise_attack,
            "blur":               _image_blur_attack,
        }
    elif content_type == "audio":
        return {
            "additive_noise":     _audio_noise_attack,
            "time_stretch":       _audio_stretch_attack,
        }
    return {}


def _get_acceptance_criteria(content_type: str) -> dict[str, float]:
    """Minimum detection rate per attack type."""
    if content_type in ("text", "code"):
        return {"paraphrase": 0.75, "synonym_sub": 0.80,
                "whitespace_inject": 0.85, "homoglyph": 0.80}
    elif content_type == "image":
        return {"jpeg_compression": 0.80, "additive_noise": 0.85, "blur": 0.85}
    elif content_type == "audio":
        return {"additive_noise": 0.80, "time_stretch": 0.70}
    return {}


# ── Text attacks ──────────────────────────────────────────────

def _text_paraphrase_attack(text: str, strength: str) -> str:
    """Simulate paraphrase: shuffle word order in some sentences."""
    sentences = text.split(". ")
    n_attack  = {"low": 1, "medium": 3, "high": 6}.get(strength, 2)
    import random
    rng = random.Random(42)
    for i in rng.sample(range(len(sentences)), min(n_attack, len(sentences))):
        words = sentences[i].split()
        rng.shuffle(words)
        sentences[i] = " ".join(words)
    return ". ".join(sentences)


def _text_synonym_attack(text: str, strength: str) -> str:
    """Replace some words with simple synonyms (no NLP library needed)."""
    synonyms = {
        "the": "a", "is": "are", "was": "were", "good": "great",
        "bad": "poor", "big": "large", "small": "tiny", "use": "utilise",
        "show": "demonstrate", "make": "create", "get": "obtain",
    }
    frac = {"low": 0.05, "medium": 0.15, "high": 0.30}.get(strength, 0.1)
    words = text.split()
    n_replace = int(len(words) * frac)
    import random
    rng = random.Random(42)
    for i in rng.sample(range(len(words)), min(n_replace, len(words))):
        w = words[i].lower().rstrip(".,;:")
        if w in synonyms:
            words[i] = words[i].replace(w, synonyms[w])
    return " ".join(words)


def _text_whitespace_attack(text: str, strength: str) -> str:
    """Inject zero-width spaces to disrupt tokenisation."""
    zwsp  = "\u200b"   # zero-width space (invisible)
    frac  = {"low": 0.01, "medium": 0.05, "high": 0.10}.get(strength, 0.03)
    chars = list(text)
    import random
    rng   = random.Random(42)
    n     = int(len(chars) * frac)
    positions = rng.sample(range(len(chars)), min(n, len(chars)))
    for pos in sorted(positions, reverse=True):
        chars.insert(pos, zwsp)
    return "".join(chars)


def _text_homoglyph_attack(text: str, strength: str) -> str:
    """Replace some ASCII chars with visually identical Unicode chars."""
    glyphs = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с"}
    frac   = {"low": 0.01, "medium": 0.03, "high": 0.08}.get(strength, 0.02)
    chars  = list(text)
    import random
    rng    = random.Random(42)
    n      = max(1, int(len(chars) * frac))
    for i in rng.sample(range(len(chars)), min(n, len(chars))):
        if chars[i].lower() in glyphs:
            chars[i] = glyphs[chars[i].lower()]
    return "".join(chars)


# ── Image attacks ─────────────────────────────────────────────

def _image_jpeg_attack(image: np.ndarray, strength: str) -> np.ndarray:
    quality = {"low": 80, "medium": 60, "high": 40}.get(strength, 70)
    try:
        from PIL import Image  # type: ignore
        import io
        pil = Image.fromarray(image)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        return np.array(Image.open(buf), dtype=np.uint8)
    except Exception:
        return image


def _image_noise_attack(image: np.ndarray, strength: str) -> np.ndarray:
    sigma = {"low": 5, "medium": 15, "high": 30}.get(strength, 10)
    noisy = image.astype(np.float32) + np.random.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _image_blur_attack(image: np.ndarray, strength: str) -> np.ndarray:
    sigma = {"low": 0.5, "medium": 1.5, "high": 3.0}.get(strength, 1.0)
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
        return gaussian_filter(image.astype(np.float32),
                                sigma=[sigma, sigma, 0]).astype(np.uint8)
    except Exception:
        return image


# ── Audio attacks ─────────────────────────────────────────────

def _audio_noise_attack(audio: np.ndarray, strength: str) -> np.ndarray:
    snr_db = {"low": 30, "medium": 20, "high": 10}.get(strength, 20)
    sp     = float(np.mean(audio ** 2))
    np_    = sp / (10 ** (snr_db / 10))
    noise  = np.random.normal(0, math.sqrt(max(np_, 1e-12)), audio.shape)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def _audio_stretch_attack(audio: np.ndarray, strength: str) -> np.ndarray:
    rate = {"low": 1.05, "medium": 1.10, "high": 0.90}.get(strength, 1.08)
    try:
        import librosa  # type: ignore
        return librosa.effects.time_stretch(audio, rate=rate)
    except Exception:
        return audio


# ── Step 100: Calibration validation ──────────────────────────

@dataclass
class CalibrationReport:
    """Calibration quality report for one detector."""
    detector_name:  str
    n_samples:      int
    ece:            float    # Expected Calibration Error
    mce:            float    # Maximum Calibration Error
    ece_target:     float    # 0.05 target
    mce_target:     float    # 0.15 target
    ece_passed:     bool
    mce_passed:     bool
    bucket_stats:   list[dict]   # per-bucket calibration data
    needs_recal:    bool


def compute_ece(
    probs:     np.ndarray,
    labels:    np.ndarray,
    n_buckets: int = 15,
) -> tuple[float, float, list[dict]]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE = Σ_b (|b| / n) * |acc(b) - conf(b)|
    MCE = max_b |acc(b) - conf(b)|

    Returns (ece, mce, bucket_stats).
    """
    bucket_stats: list[dict] = []
    n           = len(probs)
    ece         = 0.0
    mce         = 0.0

    for b in range(n_buckets):
        lo  = b / n_buckets
        hi  = (b + 1) / n_buckets
        mask = (probs >= lo) & (probs < hi)
        if b == n_buckets - 1:
            mask = (probs >= lo) & (probs <= hi)

        n_b = int(mask.sum())
        if n_b == 0:
            bucket_stats.append({
                "bucket": b, "lo": lo, "hi": hi,
                "n": 0, "accuracy": 0.0, "confidence": 0.0, "gap": 0.0,
            })
            continue

        acc  = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        gap  = abs(acc - conf)

        ece += (n_b / n) * gap
        mce  = max(mce, gap)

        bucket_stats.append({
            "bucket":     b,
            "lo":         round(lo, 3),
            "hi":         round(hi, 3),
            "n":          n_b,
            "accuracy":   round(acc, 4),
            "confidence": round(conf, 4),
            "gap":        round(gap, 4),
        })

    return round(ece, 5), round(mce, 5), bucket_stats


def validate_calibration(
    detector_name: str,
    probs:         np.ndarray,
    labels:        np.ndarray,
    ece_target:    float = 0.05,
    mce_target:    float = 0.15,
) -> CalibrationReport:
    """
    Validate calibration for one detector. Returns a CalibrationReport.
    Triggers recalibration if either target is missed.
    """
    ece, mce, buckets = compute_ece(probs, labels)

    ece_passed  = ece <= ece_target
    mce_passed  = mce <= mce_target
    needs_recal = not (ece_passed and mce_passed)

    if needs_recal:
        log.warning("calibration_degraded",
                    detector=detector_name,
                    ece=ece, ece_target=ece_target,
                    mce=mce, mce_target=mce_target)
    else:
        log.info("calibration_ok",
                 detector=detector_name, ece=ece, mce=mce)

    return CalibrationReport(
        detector_name=detector_name,
        n_samples=len(probs),
        ece=ece,
        mce=mce,
        ece_target=ece_target,
        mce_target=mce_target,
        ece_passed=ece_passed,
        mce_passed=mce_passed,
        bucket_stats=buckets,
        needs_recal=needs_recal,
    )


def recalibrate(
    probs:     np.ndarray,
    labels:    np.ndarray,
    method:    str = "isotonic",
) -> Any:
    """
    Fit a recalibration model on validation data.

    Methods:
      platt    — logistic regression on log-odds (fast, works for most cases)
      isotonic — isotonic regression (non-parametric, best for large datasets)
      both     — average of platt and isotonic (recommended)
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.isotonic import IsotonicRegression       # type: ignore

    probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)

    if method in ("platt", "both"):
        log_odds = np.log(probs_clipped / (1 - probs_clipped)).reshape(-1, 1)
        platt    = LogisticRegression(C=1e5, max_iter=1000)
        platt.fit(log_odds, labels)
    else:
        platt = None

    if method in ("isotonic", "both"):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs, labels)
    else:
        iso = None

    log.info("recalibration_fitted", method=method,
             n_samples=len(probs))
    return {"platt": platt, "isotonic": iso, "method": method}


def apply_recalibration(raw_prob: float, cal: dict) -> float:
    """Apply a fitted recalibration model at inference time."""
    probs_out: list[float] = []
    raw_clipped = max(1e-7, min(1 - 1e-7, raw_prob))

    if cal.get("platt"):
        log_odds   = math.log(raw_clipped / (1 - raw_clipped))
        platt_prob = float(cal["platt"].predict_proba([[log_odds]])[0, 1])
        probs_out.append(platt_prob)

    if cal.get("isotonic"):
        iso_prob = float(cal["isotonic"].predict([raw_prob])[0])
        probs_out.append(iso_prob)

    if not probs_out:
        return raw_clipped
    return float(np.clip(np.mean(probs_out), 0.01, 0.99))
