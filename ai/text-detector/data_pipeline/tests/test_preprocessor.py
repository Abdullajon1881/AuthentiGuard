"""
Unit tests for the dataset preprocessing pipeline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ai.text_detector.data_pipeline.preprocessing.preprocessor import (
    clean_text,
    is_valid_sample,
    normalize_unicode,
    sentence_tokenize,
    split_records,
    run_preprocessing,
)


# ── clean_text ────────────────────────────────────────────────

class TestCleanText:
    def test_strips_whitespace(self) -> None:
        assert clean_text("  hello world  ") == "hello world"

    def test_collapses_spaces(self) -> None:
        result = clean_text("hello    world")
        assert "  " not in result

    def test_removes_boilerplate(self) -> None:
        text = "Some content. Subscribe to our newsletter today."
        result = clean_text(text)
        assert "Subscribe" not in result

    def test_normalizes_unicode(self) -> None:
        # Café with decomposed é vs composed é
        composed = "caf\u00e9"
        decomposed = "cafe\u0301"
        assert normalize_unicode(composed) == normalize_unicode(decomposed)

    def test_removes_null_bytes(self) -> None:
        result = clean_text("hello\x00world")
        assert "\x00" not in result


# ── is_valid_sample ───────────────────────────────────────────

class TestIsValidSample:
    def test_valid_sample(self) -> None:
        text = " ".join([f"word{i}" for i in range(100)])
        assert is_valid_sample(text) is True

    def test_too_short(self) -> None:
        text = "This is too short."
        assert is_valid_sample(text) is False

    def test_too_long(self) -> None:
        text = " ".join(["word"] * 2000)
        assert is_valid_sample(text) is False

    def test_low_vocabulary(self) -> None:
        # Repetitive text
        text = " ".join(["the"] * 200)
        assert is_valid_sample(text) is False

    def test_mostly_non_ascii(self) -> None:
        # Arabic text (non-ASCII dominant)
        text = "مرحبا " * 100
        assert is_valid_sample(text) is False


# ── sentence_tokenize ─────────────────────────────────────────

class TestSentenceTokenize:
    def test_splits_sentences(self) -> None:
        text = "This is sentence one. This is sentence two. And a third one here."
        sentences = sentence_tokenize(text)
        assert len(sentences) >= 2

    def test_filters_short_fragments(self) -> None:
        text = "Hi. This is a proper sentence with enough words to pass the filter."
        sentences = sentence_tokenize(text)
        # "Hi." should be filtered (< 4 words)
        assert all(len(s.split()) >= 4 for s in sentences)

    def test_empty_text(self) -> None:
        assert sentence_tokenize("") == []


# ── split_records ─────────────────────────────────────────────

class TestSplitRecords:
    def _make_records(self, n_human: int, n_ai: int) -> list[dict]:
        records = []
        for i in range(n_human):
            records.append({"text": f"human text {i}", "label": 0})
        for i in range(n_ai):
            records.append({"text": f"ai text {i}", "label": 1})
        return records

    def test_correct_total(self) -> None:
        records = self._make_records(500, 500)
        splits = split_records(records)
        total = sum(len(v) for v in splits.values())
        assert total == 1000

    def test_split_names(self) -> None:
        records = self._make_records(100, 100)
        splits = split_records(records)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_train_is_largest(self) -> None:
        records = self._make_records(500, 500)
        splits = split_records(records)
        assert len(splits["train"]) > len(splits["val"])
        assert len(splits["train"]) > len(splits["test"])

    def test_stratified_both_labels_in_each_split(self) -> None:
        records = self._make_records(200, 200)
        splits = split_records(records)
        for split_name, split_records_list in splits.items():
            labels = {r["label"] for r in split_records_list}
            assert 0 in labels, f"No human samples in {split_name}"
            assert 1 in labels, f"No AI samples in {split_name}"

    def test_deterministic(self) -> None:
        records = self._make_records(100, 100)
        splits1 = split_records(records, seed=42)
        splits2 = split_records(records, seed=42)
        assert [r["label"] for r in splits1["train"]] == [r["label"] for r in splits2["train"]]


# ── run_preprocessing integration test ───────────────────────

class TestRunPreprocessing:
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Write minimal fake data
            human_records = [
                {
                    "text": " ".join([f"human word{j}" for j in range(80)]),
                    "label": 0,
                    "source": "test",
                }
                for i in range(60)
            ]
            ai_records = [
                {
                    "text": " ".join([f"ai word{j}" for j in range(80)]),
                    "label": 1,
                    "model": "test-model",
                }
                for i in range(60)
            ]

            self._write_jsonl(base / "human" / "human.jsonl", human_records)
            self._write_jsonl(base / "ai-generated" / "ai_generated.jsonl", ai_records)

            output_paths = run_preprocessing(
                datasets_dir=base,
                output_dir=base / "processed",
            )

            assert "train" in output_paths
            assert "val" in output_paths
            assert "test" in output_paths

            for split_name, path in output_paths.items():
                assert path.exists(), f"{split_name}.parquet not found"
                assert path.stat().st_size > 0

            card_path = base / "processed" / "dataset_card.json"
            assert card_path.exists()
            card = json.loads(card_path.read_text())
            assert card["total_samples"] > 0
