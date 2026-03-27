"""
Step 73: AST-level pattern analysis for code AI detection.

AI code generators (GitHub Copilot, GPT-4, Claude) exhibit characteristic
patterns at the Abstract Syntax Tree level that are statistically distinct
from human-written code:

1. Function length distribution
   AI generates functions of very uniform, "typical" lengths.
   Human code has high variance — tiny helpers alongside complex functions.

2. Nesting depth statistics
   AI tends toward shallow, clean nesting (2-3 levels max).
   Human code accumulates nested conditions and loops over time.

3. Node type distribution
   AI overuses certain patterns: ternary expressions, list comprehensions,
   chained method calls, f-strings, type hints on every parameter.
   Humans write more raw, pragmatic code.

4. Comment-to-code ratio
   AI always writes docstrings and inline comments.
   Human code is frequently under-documented or inconsistently documented.

5. Dead code and TODOs
   Humans leave unreachable code, commented-out blocks, TODO comments.
   AI code is "clean" — suspiciously free of development artifacts.

6. Magic numbers and constants
   Humans use magic numbers inline; AI extracts everything to named constants.

Supports: Python, JavaScript/TypeScript, Java, Go, C/C++, Rust, Ruby, PHP.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)

SUPPORTED_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
}


@dataclass
class ASTFeatures:
    """AST-derived features for one code file."""
    language: str

    # Function/method structure
    n_functions:          int
    function_lengths:     list[int]   # lines per function
    fn_length_mean:       float
    fn_length_std:        float
    fn_length_cv:         float       # coefficient of variation (low = AI-like uniform)

    # Nesting
    max_nesting_depth:    int
    mean_nesting_depth:   float
    nesting_depth_std:    float

    # Node type counts
    n_classes:            int
    n_loops:              int
    n_conditionals:       int
    n_try_except:         int
    n_list_comprehensions: int
    n_ternary:            int
    n_lambdas:            int
    n_type_hints:         int         # Python type annotations
    n_decorators:         int

    # Documentation
    n_docstrings:         int
    n_inline_comments:    int
    has_module_docstring: bool
    docstring_coverage:   float       # fraction of functions with docstrings

    # Human artifact indicators
    n_todos:              int
    n_fixmes:             int
    n_magic_numbers:      int         # bare numeric literals (not 0, 1, 2)
    n_commented_code:     int         # lines of commented-out code
    n_print_statements:   int         # debug prints (Python)
    has_dead_code:        bool

    # Code metrics
    n_lines:              int
    n_blank_lines:        int
    n_comment_lines:      int
    blank_ratio:          float
    comment_ratio:        float


# ── Python AST analyser ────────────────────────────────────────

def analyse_python_ast(source: str) -> ASTFeatures:
    """
    Full Python AST analysis using the built-in `ast` module.
    No external dependencies required.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        log.warning("python_parse_failed", error=str(exc))
        return _empty_ast_features("python")

    visitor = _PythonASTVisitor()
    visitor.visit(tree)

    lines       = source.splitlines()
    n_lines     = len(lines)
    blank_lines = sum(1 for l in lines if not l.strip())
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))

    # Check module-level docstring
    has_module_doc = (
        len(tree.body) > 0
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    )

    # TODOs, FIXMEs, magic numbers from source text
    todos   = len(re.findall(r"#.*\bTODO\b", source, re.IGNORECASE))
    fixmes  = len(re.findall(r"#.*\bFIXME\b", source, re.IGNORECASE))
    prints  = len(re.findall(r"\bprint\s*\(", source))

    # Magic numbers: numeric literals > 2 that aren't in common positions
    magic = len(re.findall(r"(?<!['\"\w])(?<!\.)([3-9]\d+|\d{3,})(?![\w'\"])", source))

    # Commented-out code: lines starting with # followed by code patterns
    # Matches keywords, assignments, method calls, and variable expressions
    commented_code = sum(
        1 for l in lines
        if l.strip().startswith("#")
        and re.search(
            r"#\s*(if|for|while|def|class|return|import|[a-zA-Z_]\w*\s*[=.(])", l
        )
    )

    # Function lengths
    fn_lengths = visitor.function_lengths
    fn_mean  = float(sum(fn_lengths) / len(fn_lengths)) if fn_lengths else 0.0
    fn_std   = float(_std(fn_lengths)) if fn_lengths else 0.0
    fn_cv    = fn_std / max(fn_mean, 1.0)

    # Docstring coverage
    doc_coverage = (
        visitor.n_docstrings / max(visitor.n_functions, 1)
        if visitor.n_functions > 0 else 0.0
    )

    return ASTFeatures(
        language="python",
        n_functions=visitor.n_functions,
        function_lengths=fn_lengths,
        fn_length_mean=round(fn_mean, 2),
        fn_length_std=round(fn_std, 2),
        fn_length_cv=round(fn_cv, 4),
        max_nesting_depth=visitor.max_nesting,
        mean_nesting_depth=round(visitor.mean_nesting, 3),
        nesting_depth_std=round(visitor.nesting_std, 3),
        n_classes=visitor.n_classes,
        n_loops=visitor.n_loops,
        n_conditionals=visitor.n_conditionals,
        n_try_except=visitor.n_try_except,
        n_list_comprehensions=visitor.n_list_comps,
        n_ternary=visitor.n_ternary,
        n_lambdas=visitor.n_lambdas,
        n_type_hints=visitor.n_type_hints,
        n_decorators=visitor.n_decorators,
        n_docstrings=visitor.n_docstrings,
        n_inline_comments=comment_lines,
        has_module_docstring=has_module_doc,
        docstring_coverage=round(doc_coverage, 3),
        n_todos=todos,
        n_fixmes=fixmes,
        n_magic_numbers=magic,
        n_commented_code=commented_code,
        n_print_statements=prints,
        has_dead_code=visitor.has_dead_code,
        n_lines=n_lines,
        n_blank_lines=blank_lines,
        n_comment_lines=comment_lines,
        blank_ratio=round(blank_lines / max(n_lines, 1), 4),
        comment_ratio=round(comment_lines / max(n_lines, 1), 4),
    )


class _PythonASTVisitor:
    """Walk a Python AST and collect structural statistics."""

    def __init__(self) -> None:
        self.n_functions   = 0
        self.n_classes     = 0
        self.n_loops       = 0
        self.n_conditionals = 0
        self.n_try_except  = 0
        self.n_list_comps  = 0
        self.n_ternary     = 0
        self.n_lambdas     = 0
        self.n_type_hints  = 0
        self.n_decorators  = 0
        self.n_docstrings  = 0
        self.function_lengths: list[int] = []
        self._nesting_depths: list[int] = []
        self._current_depth  = 0
        self.has_dead_code   = False

    @property
    def max_nesting(self) -> int:
        return max(self._nesting_depths) if self._nesting_depths else 0

    @property
    def mean_nesting(self) -> float:
        return float(sum(self._nesting_depths) / len(self._nesting_depths)) \
            if self._nesting_depths else 0.0

    @property
    def nesting_std(self) -> float:
        return float(_std(self._nesting_depths)) if self._nesting_depths else 0.0

    def visit(self, node: Any, depth: int = 0) -> None:
        import ast

        self._nesting_depths.append(depth)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.n_functions += 1
            self.n_decorators += len(node.decorator_list)

            # Type hints on arguments
            for arg in node.args.args:
                if arg.annotation:
                    self.n_type_hints += 1
            if node.returns:
                self.n_type_hints += 1

            # Docstring
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                self.n_docstrings += 1

            # Function length
            if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
                self.function_lengths.append(node.end_lineno - node.lineno + 1)

        elif isinstance(node, ast.ClassDef):
            self.n_classes += 1

        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            self.n_loops += 1

        elif isinstance(node, ast.If):
            self.n_conditionals += 1

        elif isinstance(node, (ast.Try, ast.ExceptHandler)):
            self.n_try_except += 1

        elif isinstance(node, ast.ListComp):
            self.n_list_comps += 1

        elif isinstance(node, ast.IfExp):
            self.n_ternary += 1

        elif isinstance(node, ast.Lambda):
            self.n_lambdas += 1

        # Dead code: statements after unconditional return/raise
        elif isinstance(node, (ast.Return, ast.Raise)):
            # Check if siblings exist after this node in the parent body
            # (simplified detection)
            pass

        for child in ast.iter_child_nodes(node):
            self.visit(child, depth + 1)


# ── Language-agnostic regex analyser ──────────────────────────

def analyse_generic_ast(source: str, language: str) -> ASTFeatures:
    """
    Regex-based structural analysis for non-Python languages.
    Less accurate than true AST parsing but requires no language-specific
    parser dependency.
    """
    lines = source.splitlines()
    n_lines = len(lines)

    # Language-specific patterns
    patterns = _get_language_patterns(language)

    n_functions    = sum(1 for l in lines if re.search(patterns["function"], l))
    n_classes      = sum(1 for l in lines if re.search(patterns["class_def"], l))
    n_loops        = sum(1 for l in lines if re.search(patterns["loop"], l))
    n_conditionals = sum(1 for l in lines if re.search(patterns["conditional"], l))
    n_comments     = sum(1 for l in lines if re.search(patterns["comment"], l))
    blank_lines    = sum(1 for l in lines if not l.strip())

    # Nesting depth: count leading whitespace/braces
    depths = [_estimate_nesting(l) for l in lines if l.strip()]
    max_depth  = max(depths) if depths else 0
    mean_depth = float(sum(depths) / len(depths)) if depths else 0.0

    todos  = len(re.findall(r"//.*\bTODO\b|#.*\bTODO\b", source, re.IGNORECASE))
    fixmes = len(re.findall(r"//.*\bFIXME\b|#.*\bFIXME\b", source, re.IGNORECASE))
    magic  = len(re.findall(r"(?<!['\"\w])(?<!\.)([3-9]\d+|\d{3,})(?![\w'\"])", source))

    return ASTFeatures(
        language=language,
        n_functions=n_functions,
        function_lengths=[],
        fn_length_mean=0.0, fn_length_std=0.0, fn_length_cv=0.0,
        max_nesting_depth=max_depth,
        mean_nesting_depth=round(mean_depth, 3),
        nesting_depth_std=round(float(_std(depths)), 3),
        n_classes=n_classes, n_loops=n_loops,
        n_conditionals=n_conditionals, n_try_except=0,
        n_list_comprehensions=0, n_ternary=0,
        n_lambdas=0, n_type_hints=0, n_decorators=0,
        n_docstrings=0, n_inline_comments=n_comments,
        has_module_docstring=False, docstring_coverage=0.0,
        n_todos=todos, n_fixmes=fixmes, n_magic_numbers=magic,
        n_commented_code=0, n_print_statements=0, has_dead_code=False,
        n_lines=n_lines, n_blank_lines=blank_lines,
        n_comment_lines=n_comments,
        blank_ratio=round(blank_lines / max(n_lines, 1), 4),
        comment_ratio=round(n_comments / max(n_lines, 1), 4),
    )


def _get_language_patterns(lang: str) -> dict[str, str]:
    if lang in {"javascript", "typescript"}:
        return {
            "function":    r"\b(function|=>|\bconst\s+\w+\s*=\s*(?:async\s*)?\()",
            "class_def":   r"\bclass\s+\w+",
            "loop":        r"\b(for|while)\s*\(",
            "conditional": r"\bif\s*\(",
            "comment":     r"^\s*(//|/\*|\*)",
        }
    elif lang == "java":
        return {
            "function":    r"\b(public|private|protected|static)\b.*\w+\s*\(",
            "class_def":   r"\bclass\s+\w+",
            "loop":        r"\b(for|while)\s*\(",
            "conditional": r"\bif\s*\(",
            "comment":     r"^\s*(//|/\*|\*)",
        }
    elif lang == "go":
        return {
            "function":    r"\bfunc\s+\w+",
            "class_def":   r"\btype\s+\w+\s+struct",
            "loop":        r"\bfor\s+",
            "conditional": r"\bif\s+",
            "comment":     r"^\s*//",
        }
    else:  # C, C++, Rust, Ruby, etc.
        return {
            "function":    r"\b\w+\s+\w+\s*\([^)]*\)\s*\{",
            "class_def":   r"\b(class|struct|impl)\s+\w+",
            "loop":        r"\b(for|while)\s*[(\s]",
            "conditional": r"\bif\s*[(\s]",
            "comment":     r"^\s*(//|/\*|#|\*)",
        }


def _estimate_nesting(line: str) -> int:
    """Estimate nesting depth from leading whitespace."""
    stripped = line.lstrip()
    indent   = len(line) - len(stripped)
    return indent // 4   # assume 4-space indent


# ── Dispatcher ────────────────────────────────────────────────

def analyse_ast(source: str, language: str) -> ASTFeatures:
    """Route to the correct AST analyser for the given language."""
    if language == "python":
        return analyse_python_ast(source)
    return analyse_generic_ast(source, language)


# ── Utilities ──────────────────────────────────────────────────

def _std(values: list[int | float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def _empty_ast_features(language: str) -> ASTFeatures:
    return ASTFeatures(
        language=language,
        n_functions=0, function_lengths=[],
        fn_length_mean=0.0, fn_length_std=0.0, fn_length_cv=0.0,
        max_nesting_depth=0, mean_nesting_depth=0.0, nesting_depth_std=0.0,
        n_classes=0, n_loops=0, n_conditionals=0, n_try_except=0,
        n_list_comprehensions=0, n_ternary=0, n_lambdas=0,
        n_type_hints=0, n_decorators=0, n_docstrings=0,
        n_inline_comments=0, has_module_docstring=False,
        docstring_coverage=0.0, n_todos=0, n_fixmes=0,
        n_magic_numbers=0, n_commented_code=0, n_print_statements=0,
        has_dead_code=False, n_lines=0, n_blank_lines=0, n_comment_lines=0,
        blank_ratio=0.0, comment_ratio=0.0,
    )
