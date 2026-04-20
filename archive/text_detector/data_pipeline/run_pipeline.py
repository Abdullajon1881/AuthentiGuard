"""
Dataset pipeline orchestrator — Step 10–15 end-to-end.

Usage:
    # Full pipeline (download + generate + preprocess):
    python -m ai.text_detector.data_pipeline.run_pipeline --all

    # Individual stages:
    python -m ai.text_detector.data_pipeline.run_pipeline --human
    python -m ai.text_detector.data_pipeline.run_pipeline --ai
    python -m ai.text_detector.data_pipeline.run_pipeline --adversarial
    python -m ai.text_detector.data_pipeline.run_pipeline --preprocess

    # Scale targets:
    python -m ai.text_detector.data_pipeline.run_pipeline --all --target 1000000
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import structlog

from .config import INITIAL_TARGET
from .sources.human_ingestion import download_human_sources
from .generators.ai_generator import generate_ai_samples
from .generators.adversarial_generator import generate_adversarial_samples
from .preprocessing.preprocessor import run_preprocessing

log = structlog.get_logger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    import logging
    import structlog

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )


def run_pipeline(
    run_human: bool = True,
    run_ai: bool = True,
    run_adversarial: bool = True,
    run_preprocess: bool = True,
    target: int = INITIAL_TARGET,
    datasets_dir: Path = Path("datasets"),
    concurrency: int = 5,
) -> None:
    """
    Full pipeline execution. Each stage is idempotent — if output files
    already exist, the stage is skipped unless forced.
    """
    start = time.time()
    log.info("pipeline_start", target=target)

    human_path = datasets_dir / "human" / "human.jsonl"
    ai_path = datasets_dir / "ai-generated" / "ai_generated.jsonl"
    adversarial_path = datasets_dir / "adversarial" / "adversarial.jsonl"

    # ── Stage 1: Human ingestion ──────────────────────────────
    if run_human:
        if human_path.exists():
            lines = sum(1 for _ in human_path.open())
            log.info("skip_human_ingestion", reason="file exists", samples=lines)
        else:
            log.info("stage_start", stage="human_ingestion")
            download_human_sources(
                target_total=int(target * 0.50),
                output_dir=datasets_dir / "human",
            )

    # ── Stage 2: AI generation ────────────────────────────────
    if run_ai:
        if ai_path.exists():
            lines = sum(1 for _ in ai_path.open())
            log.info("skip_ai_generation", reason="file exists", samples=lines)
        else:
            log.info("stage_start", stage="ai_generation")
            asyncio.run(
                generate_ai_samples(
                    target_total=int(target * 0.50),
                    output_dir=datasets_dir / "ai-generated",
                    concurrency=concurrency,
                )
            )

    # ── Stage 3: Adversarial generation ──────────────────────
    if run_adversarial:
        if adversarial_path.exists():
            lines = sum(1 for _ in adversarial_path.open())
            log.info("skip_adversarial", reason="file exists", samples=lines)
        else:
            if not ai_path.exists():
                log.warning("skip_adversarial", reason="ai_generated.jsonl not found")
            else:
                log.info("stage_start", stage="adversarial_generation")
                asyncio.run(
                    generate_adversarial_samples(
                        ai_generated_path=ai_path,
                        human_path=human_path,
                        output_dir=datasets_dir / "adversarial",
                        concurrency=max(1, concurrency // 2),
                    )
                )

    # ── Stage 4: Preprocessing ────────────────────────────────
    if run_preprocess:
        log.info("stage_start", stage="preprocessing")
        output_paths = run_preprocessing(
            datasets_dir=datasets_dir,
            output_dir=datasets_dir / "processed",
        )
        for split, path in output_paths.items():
            log.info("output", split=split, path=str(path))

    elapsed = time.time() - start
    log.info("pipeline_complete", elapsed_seconds=round(elapsed, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="AuthentiGuard dataset pipeline")

    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument("--human", action="store_true", help="Run human ingestion")
    parser.add_argument("--ai", action="store_true", help="Run AI generation")
    parser.add_argument("--adversarial", action="store_true", help="Run adversarial generation")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--target", type=int, default=INITIAL_TARGET, help="Total sample target")
    parser.add_argument("--concurrency", type=int, default=5, help="API call concurrency")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    _configure_logging(args.verbose)

    run_any = args.all or args.human or args.ai or args.adversarial or args.preprocess

    if not run_any:
        parser.print_help()
        return

    run_pipeline(
        run_human=args.all or args.human,
        run_ai=args.all or args.ai,
        run_adversarial=args.all or args.adversarial,
        run_preprocess=args.all or args.preprocess,
        target=args.target,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
