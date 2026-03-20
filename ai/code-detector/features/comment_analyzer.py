"""
Step 75: Comment style consistency analysis.

AI code generators produce highly consistent, well-formed comments:
  - Always uses correct grammar and punctuation
  - Docstrings follow a specific format (Google, NumPy, or reStructuredText)
  - Every public function has a docstring
  - Comments explain what the code does (not why)
  - No typos, no personal notes, no expletives
  - Very consistent tense and voice

Human comments show:
  - Mix of short inline notes and elaborate explanations
  - Typos, abbreviations, personal asides ("# this is a hack")
  - TODO/FIXME/HACK/XXX markers
  - Commented-out debugging code
  - Inconsistent docstring formats across the file
  - Comments that are outdated or wrong
  - Varying grammaticality and formality

Metrics:
  - Docstring format consistency (Google vs NumPy vs bare)
  - Comment capitalisation rate (AI: always capitalised)
  - Average comment length (AI: predictably medium-length)
  - Human artifact marker density
  - Professionalism score (absence of informal markers)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)


@dataclass
class CommentFeatures:
    """Comment style features for one code file."""
    # Volume
    n_comments:             int
    n_docstrings:           int
    comments_per_function:  float

    # Style consistency
    capitalisation_rate:    float   # fraction starting with uppercase
    ends_with_period_rate:  float   # fraction ending with period
    avg_comment_length:     float   # words per comment
    std_comment_length:     float

    # Docstring format
    google_style_ratio:     float   # """Args:\n  Returns:"""
    numpy_style_ratio:      float   # """Parameters\n----------"""
    bare_docstring_ratio:   float   # """Just a sentence."""
    format_consistency:     float   # how consistently one format is used

    # Human artifact markers
    has_todo:               bool
    has_fixme:              bool
    has_hack:               bool
    has_xxx:                bool
    has_debug_comments:     bool    # "debug", "test", "temp", "hack"
    informal_marker_count:  int     # total informal markers

    # AI style indicators
    ai_comment_score:       float   # composite [0,1]


# ── Comment extraction ────────────────────────────────────────

def extract_comments(source: str, language: str) -> list[str]:
    """Extract all comment strings from source, excluding docstrings."""
    if language == "python":
        return _extract_python_comments(source)
    return _extract_cstyle_comments(source)


def _extract_python_comments(source: str) -> list[str]:
    comments = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            comment = stripped[1:].strip()
            if comment:
                comments.append(comment)
    return comments


def _extract_cstyle_comments(source: str) -> list[str]:
    comments = []
    # Single-line comments
    for m in re.finditer(r"//(.+)$", source, re.MULTILINE):
        c = m.group(1).strip()
        if c:
            comments.append(c)
    # Multi-line comments (simplified)
    for m in re.finditer(r"/\*(.+?)\*/", source, re.DOTALL):
        c = m.group(1).strip()
        if c:
            comments.append(c)
    return comments


def extract_docstrings(source: str) -> list[str]:
    """Extract triple-quoted docstrings from Python source."""
    pattern = r'"""(.*?)"""|\'\'\'(.*?)\'\'\''
    docstrings = []
    for m in re.finditer(pattern, source, re.DOTALL):
        content = (m.group(1) or m.group(2) or "").strip()
        if content:
            docstrings.append(content)
    return docstrings


# ── Analysis ──────────────────────────────────────────────────

def analyse_comment_style(
    source: str,
    language: str,
    n_functions: int = 0,
) -> CommentFeatures:
    """
    Compute all comment style features for a code file.
    """
    comments   = extract_comments(source, language)
    docstrings = extract_docstrings(source) if language == "python" else []
    n_all      = len(comments) + len(docstrings)

    if not comments and not docstrings:
        return _empty_comment_features()

    # ── Volume ────────────────────────────────────────────────
    cpf = len(comments) / max(n_functions, 1)

    # ── Style consistency ─────────────────────────────────────
    cap_count = sum(1 for c in comments if c and c[0].isupper())
    cap_rate  = cap_count / max(len(comments), 1)

    period_count = sum(1 for c in comments if c.endswith("."))
    period_rate  = period_count / max(len(comments), 1)

    word_counts = [len(c.split()) for c in comments if c.strip()]
    avg_words   = float(sum(word_counts) / len(word_counts)) if word_counts else 0.0
    std_words   = float(_std(word_counts)) if word_counts else 0.0

    # ── Docstring format detection ────────────────────────────
    google_count = sum(
        1 for d in docstrings
        if re.search(r"\n\s*(Args|Returns|Raises|Note|Example):", d)
    )
    numpy_count = sum(
        1 for d in docstrings
        if re.search(r"\n\s*-{3,}", d)   # NumPy uses "------" underlines
    )
    bare_count = len(docstrings) - google_count - numpy_count

    n_docs = max(len(docstrings), 1)
    google_ratio = google_count / n_docs
    numpy_ratio  = numpy_count  / n_docs
    bare_ratio   = bare_count   / n_docs

    # Format consistency: how consistently is one format used?
    dominant = max(google_ratio, numpy_ratio, bare_ratio)
    format_consistency = dominant

    # ── Human artifact markers ────────────────────────────────
    source_lower = source.lower()
    has_todo  = bool(re.search(r"#.*\btodo\b",  source_lower))
    has_fixme = bool(re.search(r"#.*\bfixme\b", source_lower))
    has_hack  = bool(re.search(r"#.*\bhack\b",  source_lower))
    has_xxx   = bool(re.search(r"#.*\bxxx\b",   source_lower))
    has_debug = bool(re.search(
        r"#.*(debug|temp|temporary|workaround|kludge|wtf|sorry)",
        source_lower
    ))

    informal_count = sum([has_todo, has_fixme, has_hack, has_xxx, has_debug])

    # ── AI comment score ──────────────────────────────────────
    # High capitalisation rate → AI
    cap_signal = cap_rate
    # High period rate → AI (always complete sentences)
    period_signal = period_rate
    # Low comment length variance → AI (consistent, formulaic)
    cv = std_words / max(avg_words, 1.0)
    variance_signal = max(0.0, 1.0 - cv / 1.5)
    # Absence of informal markers → AI
    informal_signal = max(0.0, 1.0 - informal_count * 0.3)
    # High docstring format consistency → AI
    format_signal = format_consistency

    ai_score = (
        0.20 * cap_signal
        + 0.20 * period_signal
        + 0.20 * variance_signal
        + 0.25 * informal_signal
        + 0.15 * format_signal
    )

    return CommentFeatures(
        n_comments=len(comments),
        n_docstrings=len(docstrings),
        comments_per_function=round(cpf, 3),
        capitalisation_rate=round(cap_rate, 4),
        ends_with_period_rate=round(period_rate, 4),
        avg_comment_length=round(avg_words, 2),
        std_comment_length=round(std_words, 2),
        google_style_ratio=round(google_ratio, 4),
        numpy_style_ratio=round(numpy_ratio, 4),
        bare_docstring_ratio=round(bare_ratio, 4),
        format_consistency=round(format_consistency, 4),
        has_todo=has_todo,
        has_fixme=has_fixme,
        has_hack=has_hack,
        has_xxx=has_xxx,
        has_debug_comments=has_debug,
        informal_marker_count=informal_count,
        ai_comment_score=round(min(ai_score, 1.0), 4),
    )


def _std(values: list[int | float]) -> float:
    if len(values) < 2:
        return 0.0
    n    = len(values)
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def _empty_comment_features() -> CommentFeatures:
    return CommentFeatures(
        n_comments=0, n_docstrings=0, comments_per_function=0.0,
        capitalisation_rate=0.5, ends_with_period_rate=0.5,
        avg_comment_length=0.0, std_comment_length=0.0,
        google_style_ratio=0.0, numpy_style_ratio=0.0,
        bare_docstring_ratio=0.0, format_consistency=0.0,
        has_todo=False, has_fixme=False, has_hack=False,
        has_xxx=False, has_debug_comments=False,
        informal_marker_count=0, ai_comment_score=0.5,
    )
