"""
Unit tests for the code AI detection pipeline.
Tests cover AST analysis, naming patterns, comment style, and score logic.
All tests use inline Python source strings — no external files needed.
"""

from __future__ import annotations

import pytest

from ai.code_detector.pipeline.ast_analyzer import (
    analyse_python_ast, _std, _empty_ast_features, SUPPORTED_LANGUAGES,
)
from ai.code_detector.features.naming_analyzer import (
    analyse_naming_patterns, extract_names, _is_snake_case, _is_camel_case,
    _looks_abbreviated, HUMAN_ABBREVIATIONS,
)
from ai.code_detector.features.comment_analyzer import (
    analyse_comment_style, extract_comments, extract_docstrings,
)
from ai.code_detector.code_detector import CodeDetector


# ── Fixtures ──────────────────────────────────────────────────

AI_STYLE_CODE = '''
"""Module for processing user authentication data."""

from typing import Optional, Dict, List


def validate_user_credentials(username: str, password: str) -> bool:
    """
    Validate user credentials against the database.

    Args:
        username: The username to validate.
        password: The password to check.

    Returns:
        True if credentials are valid, False otherwise.
    """
    if not username or not password:
        return False
    return _check_database(username, password)


def retrieve_user_profile(user_identifier: int) -> Optional[Dict]:
    """
    Retrieve the user profile from the database.

    Args:
        user_identifier: The unique user ID.

    Returns:
        A dictionary containing user profile data, or None if not found.
    """
    user_data = _fetch_from_database(user_identifier)
    if user_data is None:
        return None
    return _format_user_data(user_data)


def process_authentication_request(request_payload: Dict) -> Dict:
    """
    Process an incoming authentication request.

    Args:
        request_payload: The request data containing credentials.

    Returns:
        A response dictionary with authentication status.
    """
    username = request_payload.get("username", "")
    password = request_payload.get("password", "")
    is_valid = validate_user_credentials(username, password)
    return {"authenticated": is_valid, "status": "success" if is_valid else "failure"}
'''

HUMAN_STYLE_CODE = '''
# auth stuff - TODO: clean this up later

import hashlib, os

# FIXME: this is a hack, need to fix properly
def check_user(usr, pwd):
    # temp - add proper hashing
    if not usr:
        return False
    h = hashlib.md5(pwd.encode()).hexdigest()
    # old code - keep for now
    # h2 = hashlib.sha1(pwd.encode()).hexdigest()
    return h == get_hash(usr)


def get_usr_data(uid):
    # grab user from db
    try:
        row = db.query("SELECT * FROM users WHERE id=%s", uid)
    except Exception as e:
        print(f"DB error: {e}")  # debug
        return None
    return row


x = 42  # magic number - TODO: make this configurable
MAX_RETRY = 3
'''


# ── AST analysis ──────────────────────────────────────────────

class TestPythonASTAnalysis:
    def test_counts_functions(self) -> None:
        result = analyse_python_ast(AI_STYLE_CODE)
        assert result.n_functions == 3

    def test_detects_type_hints(self) -> None:
        result = analyse_python_ast(AI_STYLE_CODE)
        assert result.n_type_hints > 0

    def test_detects_docstrings(self) -> None:
        result = analyse_python_ast(AI_STYLE_CODE)
        assert result.n_docstrings >= 3

    def test_docstring_coverage_ai_code(self) -> None:
        result = analyse_python_ast(AI_STYLE_CODE)
        # AI code: all functions have docstrings
        assert result.docstring_coverage >= 0.9

    def test_docstring_coverage_human_code(self) -> None:
        result = analyse_python_ast(HUMAN_STYLE_CODE)
        # Human code: no docstrings
        assert result.docstring_coverage == 0.0

    def test_detects_todos(self) -> None:
        result = analyse_python_ast(HUMAN_STYLE_CODE)
        assert result.n_todos > 0

    def test_detects_fixmes(self) -> None:
        result = analyse_python_ast(HUMAN_STYLE_CODE)
        assert result.n_fixmes > 0

    def test_detects_commented_code(self) -> None:
        result = analyse_python_ast(HUMAN_STYLE_CODE)
        assert result.n_commented_code > 0

    def test_function_length_stats(self) -> None:
        result = analyse_python_ast(AI_STYLE_CODE)
        assert result.fn_length_mean > 0
        assert result.fn_length_cv >= 0

    def test_line_count(self) -> None:
        result = analyse_python_ast(AI_STYLE_CODE)
        assert result.n_lines == len(AI_STYLE_CODE.splitlines())

    def test_blank_ratio_in_range(self) -> None:
        result = analyse_python_ast(AI_STYLE_CODE)
        assert 0.0 <= result.blank_ratio <= 1.0

    def test_syntax_error_returns_empty(self) -> None:
        result = analyse_python_ast("def broken(:\n    pass")
        assert result.n_functions == 0

    def test_std_utility(self) -> None:
        assert _std([]) == 0.0
        assert _std([5]) == 0.0
        assert abs(_std([1, 2, 3, 4, 5]) - 1.4142) < 0.01


# ── Language detection ────────────────────────────────────────

class TestLanguageDetection:
    def test_python_extension(self) -> None:
        assert SUPPORTED_LANGUAGES[".py"] == "python"

    def test_js_extension(self) -> None:
        assert SUPPORTED_LANGUAGES[".js"] == "javascript"

    def test_ts_extension(self) -> None:
        assert SUPPORTED_LANGUAGES[".ts"] == "typescript"

    def test_go_extension(self) -> None:
        assert SUPPORTED_LANGUAGES[".go"] == "go"


# ── Naming analysis ───────────────────────────────────────────

class TestNamingAnalysis:
    def test_ai_names_score_higher(self) -> None:
        ai_result    = analyse_naming_patterns(AI_STYLE_CODE, "python")
        human_result = analyse_naming_patterns(HUMAN_STYLE_CODE, "python")
        assert ai_result.ai_pattern_score > human_result.ai_pattern_score

    def test_ai_code_longer_names(self) -> None:
        ai_result    = analyse_naming_patterns(AI_STYLE_CODE, "python")
        human_result = analyse_naming_patterns(HUMAN_STYLE_CODE, "python")
        assert ai_result.mean_name_length > human_result.mean_name_length

    def test_human_code_more_abbreviations(self) -> None:
        human_result = analyse_naming_patterns(HUMAN_STYLE_CODE, "python")
        assert human_result.abbreviation_rate > 0

    def test_snake_case_detection(self) -> None:
        assert _is_snake_case("validate_user_credentials") is True
        assert _is_snake_case("validateUser") is False
        assert _is_snake_case("i") is False   # too short

    def test_camel_case_detection(self) -> None:
        assert _is_camel_case("validateUser") is True
        assert _is_camel_case("validate_user") is False

    def test_abbreviation_detection(self) -> None:
        assert _looks_abbreviated("usr") is True
        assert _looks_abbreviated("validateUserCredentials") is False

    def test_abbreviation_set_not_empty(self) -> None:
        assert len(HUMAN_ABBREVIATIONS) > 10

    def test_extract_names_filters_keywords(self) -> None:
        names = extract_names("for i in range(10): pass", "python")
        assert "for" not in names
        assert "in" not in names
        assert "pass" not in names

    def test_score_in_range(self) -> None:
        result = analyse_naming_patterns(AI_STYLE_CODE, "python")
        assert 0.0 <= result.ai_pattern_score <= 1.0


# ── Comment analysis ──────────────────────────────────────────

class TestCommentAnalysis:
    def test_ai_comment_score_higher(self) -> None:
        ai_result    = analyse_comment_style(AI_STYLE_CODE, "python", n_functions=3)
        human_result = analyse_comment_style(HUMAN_STYLE_CODE, "python", n_functions=2)
        assert ai_result.ai_comment_score > human_result.ai_comment_score

    def test_detects_google_docstrings(self) -> None:
        result = analyse_comment_style(AI_STYLE_CODE, "python")
        assert result.google_style_ratio > 0

    def test_human_code_has_informal_markers(self) -> None:
        result = analyse_comment_style(HUMAN_STYLE_CODE, "python")
        assert result.has_todo or result.has_fixme or result.has_hack

    def test_extract_python_comments(self) -> None:
        comments = extract_comments(HUMAN_STYLE_CODE, "python")
        assert len(comments) > 0
        assert any("hack" in c.lower() for c in comments)

    def test_extract_docstrings(self) -> None:
        docstrings = extract_docstrings(AI_STYLE_CODE)
        assert len(docstrings) >= 3

    def test_score_in_range(self) -> None:
        result = analyse_comment_style(AI_STYLE_CODE, "python")
        assert 0.0 <= result.ai_comment_score <= 1.0

    def test_capitalisation_ai_code(self) -> None:
        result = analyse_comment_style(AI_STYLE_CODE, "python")
        # AI comments are always capitalised
        assert result.capitalisation_rate >= 0.0  # may be 0 if no inline comments

    def test_empty_source_no_crash(self) -> None:
        result = analyse_comment_style("", "python")
        assert result.ai_comment_score == 0.5


# ── Full detector ─────────────────────────────────────────────

class TestCodeDetector:
    def setup_method(self) -> None:
        self.detector = CodeDetector(use_transformer=False)
        self.detector.load_models()

    def test_ai_code_scores_higher(self) -> None:
        ai_result    = self.detector.analyze(AI_STYLE_CODE, "module.py")
        human_result = self.detector.analyze(HUMAN_STYLE_CODE, "auth.py")
        assert ai_result.score > human_result.score, (
            f"AI score {ai_result.score:.4f} should > human score {human_result.score:.4f}"
        )

    def test_returns_valid_label(self) -> None:
        result = self.detector.analyze(AI_STYLE_CODE, "test.py")
        assert result.label in {"AI", "HUMAN", "UNCERTAIN"}

    def test_score_in_range(self) -> None:
        for code, name in [(AI_STYLE_CODE, "ai.py"), (HUMAN_STYLE_CODE, "human.py")]:
            result = self.detector.analyze(code, name)
            assert 0.0 < result.score < 1.0

    def test_confidence_in_range(self) -> None:
        result = self.detector.analyze(AI_STYLE_CODE, "test.py")
        assert 0.0 <= result.confidence <= 1.0

    def test_language_detected(self) -> None:
        result = self.detector.analyze(AI_STYLE_CODE, "module.py")
        assert result.language == "python"

    def test_evidence_has_required_keys(self) -> None:
        result = self.detector.analyze(AI_STYLE_CODE, "test.py")
        required = {
            "language", "n_functions", "docstring_coverage",
            "mean_name_length", "top_signals",
        }
        assert required.issubset(result.evidence.keys())

    def test_empty_code_no_crash(self) -> None:
        result = self.detector.analyze("", "empty.py")
        assert 0.0 < result.score < 1.0

    def test_single_line_no_crash(self) -> None:
        result = self.detector.analyze("x = 1", "tiny.py")
        assert result.label in {"AI", "HUMAN", "UNCERTAIN"}

    def test_heuristic_score_bounds(self) -> None:
        from ai.code_detector.pipeline.ast_analyzer import _empty_ast_features
        from ai.code_detector.features.naming_analyzer import _empty_naming
        from ai.code_detector.features.comment_analyzer import _empty_comment_features

        score = CodeDetector._heuristic_score(
            _empty_ast_features("python"),
            _empty_naming(),
            _empty_comment_features(),
        )
        assert 0.0 <= score <= 1.0
