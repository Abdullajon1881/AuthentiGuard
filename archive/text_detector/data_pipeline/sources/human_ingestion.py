"""
Step 10: Download and organize human text from Wikipedia, Reddit, arXiv, news.
Streams from HuggingFace datasets to stay memory-efficient at scale.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterator

import structlog

from ..config import HUMAN_SOURCES, HumanSource, INITIAL_TARGET, HUMAN_RATIO

log = structlog.get_logger(__name__)

OUTPUT_DIR = Path("datasets/human")


def _count_words(text: str) -> int:
    return len(text.split())


def _clean_text(text: str) -> str:
    """Basic cleaning: strip excess whitespace, remove null bytes."""
    text = text.replace("\x00", "")
    text = " ".join(text.split())
    return text.strip()


def _stream_source(source: HumanSource) -> Iterator[str]:
    """Stream text samples from a single HuggingFace source."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc

    log.info("streaming_source", source=source.name, dataset=source.hf_dataset)

    ds = load_dataset(
        source.hf_dataset,
        source.hf_config,
        split=source.hf_split,
        streaming=True,
        trust_remote_code=True,
    )

    for row in ds:
        text = row.get(source.text_column, "")
        if not isinstance(text, str):
            continue
        text = _clean_text(text)
        word_count = _count_words(text)
        if source.min_words <= word_count <= source.max_words:
            yield text


def download_human_sources(
    target_total: int = int(INITIAL_TARGET * HUMAN_RATIO),
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Download human text from all configured sources.

    Distributes samples evenly across sources, streams to disk
    in JSONL format. Returns path to output file.

    Args:
        target_total: Total human samples to collect.
        output_dir:   Where to write the JSONL file.

    Returns:
        Path to the written human.jsonl file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "human.jsonl"

    per_source = target_total // len(HUMAN_SOURCES)
    log.info(
        "starting_human_download",
        target_total=target_total,
        per_source=per_source,
        sources=[s.name for s in HUMAN_SOURCES],
    )

    total_written = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for source in HUMAN_SOURCES:
            source_count = 0
            for text in _stream_source(source):
                if source_count >= per_source:
                    break
                record = {
                    "text": text,
                    "label": 0,           # 0 = human
                    "source": source.name,
                    "word_count": _count_words(text),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                source_count += 1
                total_written += 1

                if source_count % 1000 == 0:
                    log.info(
                        "progress",
                        source=source.name,
                        collected=source_count,
                        target=per_source,
                    )

            log.info("source_complete", source=source.name, collected=source_count)

    log.info("human_download_complete", total=total_written, path=str(out_path))
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_human_sources()
