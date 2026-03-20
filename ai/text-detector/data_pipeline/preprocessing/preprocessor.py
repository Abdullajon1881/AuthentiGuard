"""
Step 15: Preprocessing pipeline — tokenization, cleaning, normalization,
and train/val/test splitting for all dataset splits.

Produces final .parquet files that the model training pipelines consume.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator

import structlog

from ..config import SPLIT_RATIOS

log = structlog.get_logger(__name__)


# ── Text cleaning ──────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """NFC normalize and strip non-printable control characters."""
    text = unicodedata.normalize("NFC", text)
    # Remove control chars except newline/tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def fix_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines, strip edges."""
    text = re.sub(r"\n{3,}", "\n\n", text)      # max 2 consecutive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)       # collapse spaces
    return text.strip()


def remove_boilerplate(text: str) -> str:
    """Strip common web boilerplate patterns."""
    patterns = [
        r"Subscribe to our newsletter.*",
        r"Click here to read more.*",
        r"Share this article.*",
        r"\[edit\]",
        r"Retrieved from.*",
        r"This article.*stub.*",
        r"===+",
        r"---+",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single text sample."""
    text = normalize_unicode(text)
    text = remove_boilerplate(text)
    text = fix_whitespace(text)
    return text


def is_valid_sample(
    text: str,
    min_words: int = 50,
    max_words: int = 1200,
    min_unique_words: int = 20,
) -> bool:
    """Quality filter — reject low-quality samples."""
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False

    # Reject if vocabulary is too small (repeated text, spam)
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    if unique_ratio < 0.25:
        return False

    # Reject if mostly non-ASCII (wrong language / encoding artifact)
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio < 0.70:
        return False

    return True


# ── Sentence tokenization ──────────────────────────────────────

def sentence_tokenize(text: str) -> list[str]:
    """
    Split text into sentences for per-sentence scoring at inference time.
    Uses a simple rule-based splitter — fast, no NLTK download required.
    """
    # Split on .!? followed by space+capital
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    # Filter very short fragments (< 4 words)
    return [s.strip() for s in sentences if len(s.split()) >= 4]


# ── Dataset splitting ──────────────────────────────────────────

def split_records(
    records: list[dict],
    ratios: dict[str, float] = SPLIT_RATIOS,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Deterministic train/val/test split, stratified by label.
    """
    import random
    rng = random.Random(seed)

    human = [r for r in records if r["label"] == 0]
    ai    = [r for r in records if r["label"] == 1]

    splits: dict[str, list[dict]] = {k: [] for k in ratios}

    for subset in (human, ai):
        rng.shuffle(subset)
        n = len(subset)
        cursor = 0
        for split_name, ratio in ratios.items():
            end = cursor + round(n * ratio)
            splits[split_name].extend(subset[cursor:end])
            cursor = end
        # any leftover goes to train
        splits["train"].extend(subset[cursor:])

    for split_name in splits:
        rng.shuffle(splits[split_name])

    return splits


# ── Main pipeline ─────────────────────────────────────────────

def _load_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        log.warning("file_not_found", path=str(path))
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("json_parse_error", error=str(exc))


def run_preprocessing(
    datasets_dir: Path = Path("datasets"),
    output_dir: Path = Path("datasets/processed"),
) -> dict[str, Path]:
    """
    Full preprocessing pipeline:
    1. Load all JSONL files from human/, ai-generated/, adversarial/
    2. Clean + validate each sample
    3. Add sentence tokenization
    4. Stratified train/val/test split
    5. Write final .parquet files per split

    Returns:
        Dict mapping split name → output parquet path.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pip install pandas pyarrow") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all sources ──
    all_records: list[dict] = []

    sources = [
        datasets_dir / "human" / "human.jsonl",
        datasets_dir / "ai-generated" / "ai_generated.jsonl",
        datasets_dir / "adversarial" / "adversarial.jsonl",
    ]

    for src_path in sources:
        raw_count = 0
        kept_count = 0
        for record in _load_jsonl(src_path):
            raw_count += 1
            text = record.get("text", "")
            text = clean_text(text)
            if not is_valid_sample(text):
                continue

            record["text"] = text
            record["sentences"] = sentence_tokenize(text)
            record["n_sentences"] = len(record["sentences"])
            record["word_count"] = len(text.split())
            all_records.append(record)
            kept_count += 1

        log.info(
            "source_loaded",
            path=str(src_path),
            raw=raw_count,
            kept=kept_count,
            drop_rate=f"{(raw_count - kept_count) / max(raw_count, 1):.1%}",
        )

    log.info("total_samples", count=len(all_records))

    if not all_records:
        log.warning("no_samples_found — run ingestion first")
        return {}

    # ── Split ──
    splits = split_records(all_records)

    for split_name, split_size in {k: len(v) for k, v in splits.items()}.items():
        label_counts = {}
        for r in splits[split_name]:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        log.info(
            "split_stats",
            split=split_name,
            total=split_size,
            human=label_counts.get(0, 0),
            ai=label_counts.get(1, 0),
        )

    # ── Write parquet ──
    output_paths: dict[str, Path] = {}
    for split_name, records in splits.items():
        df = pd.DataFrame(records)
        out_path = output_dir / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        output_paths[split_name] = out_path
        log.info("wrote_split", split=split_name, rows=len(df), path=str(out_path))

    # ── Write dataset card ──
    card = {
        "total_samples": len(all_records),
        "splits": {k: len(v) for k, v in splits.items()},
        "label_distribution": {
            "human": sum(1 for r in all_records if r["label"] == 0),
            "ai": sum(1 for r in all_records if r["label"] == 1),
        },
        "sources": [str(s) for s in sources],
    }
    card_path = output_dir / "dataset_card.json"
    with card_path.open("w") as f:
        json.dump(card, f, indent=2)

    log.info("preprocessing_complete", output_dir=str(output_dir))
    return output_paths


if __name__ == "__main__":
    run_preprocessing()
