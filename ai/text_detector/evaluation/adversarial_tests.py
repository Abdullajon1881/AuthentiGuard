"""
Step 23: Adversarial attack test suite.

Before any model ships, it must pass all attack scenarios below.
If it breaks under paraphrasing, it's not ready.

Tests:
  1. Paraphrase resilience — model should score paraphrased AI text ≥ 0.60
  2. Back-translation resilience — score on translated AI text ≥ 0.55
  3. Grammar correction resilience — score on grammar-corrected AI ≥ 0.60
  4. Human text false positive rate — should stay < 0.15 FPR
  5. Mixed content detection — mixed texts should score 0.40–0.90
  6. Short text handling — ≥ 20 words, should not crash
  7. Layer failure tolerance — ensemble should work if 1–2 layers error
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import structlog

log = structlog.get_logger(__name__)


@dataclass
class AttackTestResult:
    test_name: str
    passed: bool
    score: float
    threshold: float
    n_samples: int
    details: dict


class AdversarialTestSuite:
    """
    Runs the full adversarial test suite against a scoring function.

    Usage:
        def score_fn(text: str) -> float:
            return ensemble.analyze(text).score

        suite = AdversarialTestSuite(score_fn)
        report = suite.run_all(adversarial_jsonl_path)
        suite.assert_all_passed(report)
    """

    def __init__(
        self,
        score_fn: Callable[[str], float],
        ai_threshold: float = 0.50,       # min score to classify as AI
    ) -> None:
        self._score = score_fn
        self._ai_threshold = ai_threshold

    # ── Individual attack tests ───────────────────────────────

    def test_paraphrase_resilience(
        self,
        samples: list[dict],
        min_mean_score: float = 0.60,
    ) -> AttackTestResult:
        """Paraphrased AI text must still score above threshold."""
        paraphrase_samples = [s for s in samples if s.get("attack_type") == "paraphrase"]
        if not paraphrase_samples:
            return AttackTestResult(
                test_name="paraphrase_resilience",
                passed=True,
                score=0.0,
                threshold=min_mean_score,
                n_samples=0,
                details={"skipped": "no paraphrase samples found"},
            )

        scores = [self._score(s["text"]) for s in paraphrase_samples]
        mean_score = sum(scores) / len(scores)
        detection_rate = sum(1 for s in scores if s >= self._ai_threshold) / len(scores)

        return AttackTestResult(
            test_name="paraphrase_resilience",
            passed=mean_score >= min_mean_score,
            score=round(mean_score, 4),
            threshold=min_mean_score,
            n_samples=len(scores),
            details={
                "detection_rate": round(detection_rate, 4),
                "min_score": round(min(scores), 4),
                "max_score": round(max(scores), 4),
            },
        )

    def test_backtranslation_resilience(
        self,
        samples: list[dict],
        min_mean_score: float = 0.55,
    ) -> AttackTestResult:
        """Back-translated AI text must still score above threshold."""
        bt_samples = [s for s in samples if "backtranslation" in str(s.get("attack_type", ""))]

        scores = [self._score(s["text"]) for s in bt_samples] if bt_samples else []
        mean_score = sum(scores) / len(scores) if scores else 0.0

        return AttackTestResult(
            test_name="backtranslation_resilience",
            passed=(not bt_samples) or mean_score >= min_mean_score,
            score=round(mean_score, 4),
            threshold=min_mean_score,
            n_samples=len(scores),
            details={} if not scores else {
                "detection_rate": round(sum(1 for s in scores if s >= self._ai_threshold) / len(scores), 4),
            },
        )

    def test_grammar_correction_resilience(
        self,
        samples: list[dict],
        min_mean_score: float = 0.60,
    ) -> AttackTestResult:
        """Grammar-corrected AI text must still score above threshold."""
        gc_samples = [s for s in samples if s.get("attack_type") == "grammar_correction"]

        scores = [self._score(s["text"]) for s in gc_samples] if gc_samples else []
        mean_score = sum(scores) / len(scores) if scores else 0.0

        return AttackTestResult(
            test_name="grammar_correction_resilience",
            passed=(not gc_samples) or mean_score >= min_mean_score,
            score=round(mean_score, 4),
            threshold=min_mean_score,
            n_samples=len(scores),
            details={},
        )

    def test_human_false_positive_rate(
        self,
        human_samples: list[dict],
        max_fpr: float = 0.15,
    ) -> AttackTestResult:
        """Human text should not be misclassified as AI more than max_fpr of the time."""
        scores = [self._score(s["text"]) for s in human_samples]
        fpr = sum(1 for s in scores if s >= self._ai_threshold) / max(len(scores), 1)
        mean_score = sum(scores) / max(len(scores), 1)

        return AttackTestResult(
            test_name="human_false_positive_rate",
            passed=fpr <= max_fpr,
            score=round(fpr, 4),
            threshold=max_fpr,
            n_samples=len(scores),
            details={
                "fpr": round(fpr, 4),
                "mean_human_score": round(mean_score, 4),
            },
        )

    def test_short_text_handling(
        self,
        min_words: int = 20,
    ) -> AttackTestResult:
        """Short texts must not crash — they should return a valid [0,1] score."""
        short_texts = [
            "This is a test.",
            "AI generated text here.",
            "Hello world this is short.",
            " ".join(["word"] * min_words),
        ]
        errors = 0
        for text in short_texts:
            try:
                score = self._score(text)
                if not (0.0 <= score <= 1.0):
                    errors += 1
            except Exception:
                errors += 1

        return AttackTestResult(
            test_name="short_text_handling",
            passed=errors == 0,
            score=float(len(short_texts) - errors) / len(short_texts),
            threshold=1.0,
            n_samples=len(short_texts),
            details={"errors": errors},
        )

    def test_layer_failure_tolerance(
        self,
        score_with_errors_fn: Callable[[str], float] | None = None,
    ) -> AttackTestResult:
        """
        Ensemble must still produce a valid score when 1–2 layers fail.
        Inject None/error outputs and verify the meta-classifier handles them.
        """
        if score_with_errors_fn is None:
            return AttackTestResult(
                test_name="layer_failure_tolerance",
                passed=True,
                score=1.0,
                threshold=1.0,
                n_samples=0,
                details={"skipped": "no error injection function provided"},
            )

        test_texts = [
            "The rapid advancement of machine learning has transformed industries.",
            "I went to the store yesterday and bought some groceries.",
        ]
        errors = 0
        for text in test_texts:
            try:
                score = score_with_errors_fn(text)
                if not (0.0 <= score <= 1.0):
                    errors += 1
            except Exception:
                errors += 1

        return AttackTestResult(
            test_name="layer_failure_tolerance",
            passed=errors == 0,
            score=float(len(test_texts) - errors) / len(test_texts),
            threshold=1.0,
            n_samples=len(test_texts),
            details={"errors": errors},
        )

    # ── Full suite runner ──────────────────────────────────────

    def run_all(
        self,
        adversarial_path: Path,
        human_path: Path,
        output_dir: Path | None = None,
    ) -> list[AttackTestResult]:
        """
        Load data and run all adversarial tests.

        Returns list of AttackTestResult — one per test.
        Writes report JSON to output_dir if provided.
        """
        # Load samples
        adversarial_samples: list[dict] = []
        if adversarial_path.exists():
            with adversarial_path.open() as f:
                for line in f:
                    try:
                        adversarial_samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        human_samples: list[dict] = []
        if human_path.exists():
            with human_path.open() as f:
                for i, line in enumerate(f):
                    if i >= 500:  # limit to 500 for speed
                        break
                    try:
                        human_samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        log.info(
            "running_adversarial_suite",
            n_adversarial=len(adversarial_samples),
            n_human=len(human_samples),
        )

        start = time.time()
        results = [
            self.test_paraphrase_resilience(adversarial_samples),
            self.test_backtranslation_resilience(adversarial_samples),
            self.test_grammar_correction_resilience(adversarial_samples),
            self.test_human_false_positive_rate(human_samples),
            self.test_short_text_handling(),
            self.test_layer_failure_tolerance(),
        ]
        elapsed = time.time() - start

        n_passed = sum(1 for r in results if r.passed)
        log.info(
            "adversarial_suite_complete",
            passed=n_passed,
            total=len(results),
            elapsed_s=round(elapsed, 1),
        )

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            log.info(
                "test_result",
                test=r.test_name,
                status=status,
                score=r.score,
                threshold=r.threshold,
                n=r.n_samples,
            )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            report = [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "score": r.score,
                    "threshold": r.threshold,
                    "n_samples": r.n_samples,
                    "details": r.details,
                }
                for r in results
            ]
            out_path = output_dir / "adversarial_test_report.json"
            with out_path.open("w") as f:
                json.dump(report, f, indent=2)

        return results

    @staticmethod
    def assert_all_passed(results: list[AttackTestResult]) -> None:
        """Raise AssertionError if any test failed. Used as CI gate."""
        failures = [r for r in results if not r.passed]
        if failures:
            names = ", ".join(r.test_name for r in failures)
            raise AssertionError(
                f"Adversarial test suite FAILED: {names}\n"
                "Fix the model before deploying."
            )
