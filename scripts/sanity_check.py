#!/usr/bin/env python3
"""
End-to-end sanity check for the AuthentiGuard text detector.

Runs detection on two short fixtures (one AI-like, one human-like) and
asserts that the pipeline returns well-formed results. No network, no
database, no worker. Targets <10s on CPU after models are cached.

Usage:
    python scripts/sanity_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the repo root importable regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai.text_detector.pipeline import analyze, MODEL_VERSION  # noqa: E402

# Long enough (>50 words) to bypass the G1 short-text gate.
AI_LIKE = (
    "Furthermore, it is important to note that the implementation leverages a "
    "comprehensive and robust framework to facilitate seamless integration "
    "across multiple paradigms. Moreover, the solution delves into nuanced "
    "aspects of the problem, providing a multifaceted perspective that "
    "ultimately enables stakeholders to make well-informed decisions. "
    "Additionally, the approach consequently demonstrates significant improvements."
)

HUMAN_LIKE = (
    "So I was thinking about this the other day, you know, how weird it is "
    "that we all just kind of accept the things we do. Like, basically, we "
    "wake up, do our stuff, and then go to sleep — and it all just feels "
    "pretty normal. Actually, when you really think about it, though, it's "
    "kind of strange. Anyway, I guess that's just how life works, I mean, "
    "there's not much you can do about it. Well, whatever."
)


def _assert_result(name: str, text: str) -> None:
    t0 = time.monotonic()
    result = analyze(text)
    dt = time.monotonic() - t0

    assert 0.0 <= result.score <= 1.0, f"{name}: score out of range: {result.score}"
    assert result.label in {"AI", "HUMAN", "UNCERTAIN"}, f"{name}: bad label {result.label}"
    assert 0.0 <= result.confidence <= 1.0, f"{name}: confidence out of range"
    assert "product" in result.evidence_summary, f"{name}: missing product block"
    assert result.evidence_summary["product"]["model_version"] == MODEL_VERSION

    print(
        f"  {name:12s}  label={result.label:10s}  score={result.score:.4f}  "
        f"conf={result.confidence:.3f}  dt={dt * 1000:.0f}ms"
    )


def main() -> int:
    print(f"AuthentiGuard sanity check — model={MODEL_VERSION}")
    print("-" * 68)

    t_start = time.monotonic()
    _assert_result("ai-like",    AI_LIKE)
    _assert_result("human-like", HUMAN_LIKE)
    total = time.monotonic() - t_start

    print("-" * 68)
    print(f"OK  ({total:.2f}s total including first-request model warm-up)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
