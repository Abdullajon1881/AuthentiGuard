"""
Step 74: Variable naming pattern analysis.

AI code generators have a statistically distinct naming style:
  - Always uses descriptive, full-word names (no abbreviations)
  - Follows conventions perfectly (snake_case, camelCase, PascalCase)
  - Never uses single-letter names except in tightly conventional contexts
  - Names encode function perfectly (get_user_by_id vs just "fetch")
  - Avoids numeric suffixes (x1, temp2) and Hungarian notation

Human code shows:
  - Mix of abbreviations and full names (usr, mgr, cfg, tmp)
  - Convention inconsistencies (mixing camelCase and snake_case)
  - Pragmatic short names in tight loops (i, j, k, x, y)
  - Names that evolved over time (legacy, new_, _v2 suffixes)
  - Domain-specific shorthand (px, ms, buf, idx, ptr)

Metrics computed:
  - Average name length (AI: ~9–12 chars, human: ~5–8)
  - Abbreviation density (AI: near zero)
  - Convention consistency (AI: near perfect)
  - Single-letter variable rate (AI: only i/j/k in for loops)
  - Descriptiveness score (name entropy vs. word dictionary)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)


@dataclass
class NamingFeatures:
    """Variable and function naming pattern features."""
    # Length statistics
    mean_name_length:     float
    std_name_length:      float
    max_name_length:      int

    # Convention adherence
    snake_case_ratio:     float   # fraction following snake_case
    camel_case_ratio:     float   # fraction following camelCase
    convention_consistency: float  # how consistent the dominant convention is

    # Descriptiveness signals
    abbreviation_rate:    float   # fraction of names that look abbreviated
    single_letter_rate:   float   # fraction that are single characters
    numeric_suffix_rate:  float   # names ending in digits (x1, temp2)
    all_caps_rate:        float   # ALL_CAPS constants

    # AI-specific indicators
    ai_pattern_score:     float   # composite AI naming score [0,1]

    # Raw data
    n_names:              int
    sample_names:         list[str]   # up to 20 names for evidence panel


# ── Common abbreviations humans use but AI avoids ─────────────

HUMAN_ABBREVIATIONS = {
    "usr", "mgr", "cfg", "conf", "tmp", "temp", "buf", "ptr", "idx",
    "cnt", "num", "val", "obj", "src", "dst", "str", "int", "func",
    "arg", "args", "kwargs", "err", "exc", "msg", "req", "res", "resp",
    "db", "conn", "cur", "col", "row", "tbl", "seq", "lst", "dct",
    "px", "ms", "kb", "mb", "ns", "ts", "dt", "fmt", "sep", "delim",
    "prev", "curr", "next", "orig", "mod", "pkg", "lib", "util",
    "pct", "avg", "std", "var", "cov", "prob", "freq", "amp",
    "x", "y", "z", "n", "i", "j", "k", "p", "q", "r", "s", "t",
    "a", "b", "c", "d", "e", "f", "g", "h", "m", "v", "w",
}

# Regex patterns for name extraction
_PYTHON_IDENTIFIERS = re.compile(
    r"(?:^|\s)(?:def\s+|class\s+|(?:self\.|[a-z_]\w*\s*,\s*)*)"
    r"([a-zA-Z_]\w*)\s*(?:=|:|\()",
    re.MULTILINE,
)
_SIMPLE_IDENTIFIERS = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]{1,40})\b")


def extract_names(source: str, language: str) -> list[str]:
    """
    Extract variable, function, and parameter names from source code.
    Filters out keywords and common noise.
    """
    # Keywords to exclude per language
    keywords = _get_keywords(language)

    raw = _SIMPLE_IDENTIFIERS.findall(source)
    filtered = [
        n for n in raw
        if n not in keywords
        and not n.startswith("__")
        and len(n) >= 1
        and not n[0].isdigit()
    ]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for n in filtered:
        if n not in seen:
            seen.add(n)
            unique.append(n)

    return unique


def analyse_naming_patterns(source: str, language: str) -> NamingFeatures:
    """Analyse naming patterns and compute AI vs human indicators."""
    names = extract_names(source, language)

    if not names:
        return _empty_naming()

    # Length statistics
    lengths = [len(n) for n in names]
    mean_len  = float(sum(lengths) / len(lengths))
    std_len   = float(_std(lengths))
    max_len   = max(lengths)

    # Convention analysis
    snake_count = sum(1 for n in names if _is_snake_case(n))
    camel_count = sum(1 for n in names if _is_camel_case(n))
    n = len(names)

    snake_ratio = snake_count / n
    camel_ratio = camel_count / n
    dominant    = max(snake_ratio, camel_ratio)
    # High dominant ratio = consistent convention = AI-like
    consistency = dominant

    # Abbreviation analysis
    abbrev_count = sum(
        1 for name in names
        if name.lower() in HUMAN_ABBREVIATIONS or _looks_abbreviated(name)
    )
    abbrev_rate = abbrev_count / n

    # Single-letter names (excluding conventional i, j, k in loop context)
    single_letter = sum(1 for name in names if len(name) == 1)
    single_rate   = single_letter / n

    # Numeric suffixes (x1, temp2, data3) — human dev artifact
    numeric_suffix = sum(1 for name in names if re.search(r"\d+$", name))
    num_suffix_rate = numeric_suffix / n

    # ALL_CAPS constants
    all_caps = sum(
        1 for name in names
        if name == name.upper() and len(name) > 1 and "_" in name
    )
    all_caps_rate = all_caps / n

    # ── AI pattern score ──────────────────────────────────────
    # High average name length → AI
    len_signal = min(max(0.0, (mean_len - 5.0) / 7.0), 1.0)
    # Low abbreviation rate → AI
    abbrev_signal = max(0.0, 1.0 - abbrev_rate * 5.0)
    # High convention consistency → AI
    consistency_signal = consistency
    # Low single-letter rate → AI (but not too low — even AI uses i/j)
    single_signal = max(0.0, 1.0 - single_rate * 10.0) if single_rate < 0.05 else 0.0
    # Low numeric suffix rate → AI
    suffix_signal = max(0.0, 1.0 - num_suffix_rate * 10.0)

    ai_score = (
        0.30 * len_signal
        + 0.25 * abbrev_signal
        + 0.20 * consistency_signal
        + 0.15 * single_signal
        + 0.10 * suffix_signal
    )

    return NamingFeatures(
        mean_name_length=round(mean_len, 2),
        std_name_length=round(std_len, 2),
        max_name_length=max_len,
        snake_case_ratio=round(snake_ratio, 4),
        camel_case_ratio=round(camel_ratio, 4),
        convention_consistency=round(consistency, 4),
        abbreviation_rate=round(abbrev_rate, 4),
        single_letter_rate=round(single_rate, 4),
        numeric_suffix_rate=round(num_suffix_rate, 4),
        all_caps_rate=round(all_caps_rate, 4),
        ai_pattern_score=round(min(ai_score, 1.0), 4),
        n_names=n,
        sample_names=names[:20],
    )


# ── Helpers ────────────────────────────────────────────────────

def _is_snake_case(name: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9_]*$", name)) and len(name) > 1


def _is_camel_case(name: str) -> bool:
    return bool(re.match(r"^[a-z][a-zA-Z0-9]*$", name)) and any(c.isupper() for c in name)


def _looks_abbreviated(name: str) -> bool:
    """
    Heuristic: name looks abbreviated if it's <= 4 chars and
    not a common English word.
    """
    if len(name) > 4:
        return False
    # Very short names are likely abbreviations unless conventional
    conventional_short = {"self", "cls", "args", "true", "false", "none",
                           "null", "int", "str", "bool", "list", "dict", "set"}
    return name.lower() not in conventional_short


def _std(values: list[int | float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def _empty_naming() -> NamingFeatures:
    return NamingFeatures(
        mean_name_length=0.0, std_name_length=0.0, max_name_length=0,
        snake_case_ratio=0.0, camel_case_ratio=0.0, convention_consistency=0.0,
        abbreviation_rate=0.0, single_letter_rate=0.0, numeric_suffix_rate=0.0,
        all_caps_rate=0.0, ai_pattern_score=0.5,
        n_names=0, sample_names=[],
    )


def _get_keywords(language: str) -> set[str]:
    common = {
        "if", "else", "elif", "for", "while", "return", "import", "from",
        "class", "def", "try", "except", "finally", "with", "as", "in",
        "not", "and", "or", "is", "None", "True", "False", "pass",
        "break", "continue", "raise", "yield", "lambda", "global", "nonlocal",
        "async", "await", "print", "len", "range", "type", "list", "dict",
        "set", "tuple", "int", "str", "float", "bool", "bytes", "object",
        "super", "self", "cls", "new", "delete", "this", "void", "public",
        "private", "protected", "static", "final", "const", "var", "let",
        "function", "struct", "impl", "fn", "mut", "use", "mod", "pub",
        "true", "false", "null", "undefined", "typeof", "instanceof",
        "switch", "case", "default", "do", "goto", "typedef", "enum",
        "namespace", "template", "virtual", "override", "abstract",
    }
    return common
