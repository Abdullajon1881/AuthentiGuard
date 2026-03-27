"""
Three-phase training for video deepfake detector.

Phase 1 — Pretrain: ImageNet pretrained weights (via timm)
Phase 2 — Fine-tune: FaceForensics++, DFDC, Celeb-DF, DeeperForensics
Phase 3 — Adversarial hardening: compressed video, cropped frames, augmentation

Usage:
    python -m ai.video_detector.training.train \
        --phase 2 \
        --data-dir datasets/video \
        --output-dir ai/video_detector/checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train video deepfake detector")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True,
                        help="Training phase (1=pretrain, 2=finetune, 3=adversarial)")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/video"),
                        help="Directory containing video datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("ai/video_detector/checkpoints"),
                        help="Directory for saving checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (defaults: phase1=1e-4, phase2=5e-5, phase3=1e-5)")

    args = parser.parse_args()

    log.info(
        "video_training_start",
        phase=args.phase,
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    from ..models.classifier import train_video_phase

    checkpoint_path = train_video_phase(
        phase=args.phase,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    log.info("video_training_complete", checkpoint=str(checkpoint_path))


if __name__ == "__main__":
    main()
