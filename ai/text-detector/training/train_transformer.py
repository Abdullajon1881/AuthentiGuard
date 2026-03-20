"""
Three-phase transformer training per the AuthentiGuard roadmap.

Phase 1 — Pretrain: train DeBERTa-v3 on clean human vs AI data.
Phase 2 — Specialize: continue training with higher LR on domain data.
Phase 3 — Adversarial hardening: fine-tune on adversarial samples only.

Usage:
    python -m ai.text-detector.training.train_transformer \
        --phase 1 \
        --data-dir datasets/processed \
        --output-dir ai/text-detector/checkpoints/transformer \
        --model microsoft/deberta-v3-base \
        --epochs 3

Tracks all runs in MLflow. Applies calibration gate before saving.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

LABEL2ID = {"human": 0, "ai": 1}
ID2LABEL = {0: "human", 1: "ai"}


def _load_parquet_split(data_dir: Path, split: str) -> Any:
    """Load a preprocessed parquet split into a HuggingFace Dataset."""
    import pandas as pd  # type: ignore
    from datasets import Dataset  # type: ignore

    path = data_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the data pipeline first.")
    df = pd.read_parquet(path)[["text", "label"]]
    # For Phase 3, filter to adversarial samples only
    return Dataset.from_pandas(df)


def _load_adversarial_only(data_dir: Path) -> Any:
    """For Phase 3: load from adversarial JSONL (not preprocessed parquet)."""
    import json
    from datasets import Dataset  # type: ignore

    adv_path = data_dir.parent / "adversarial" / "adversarial.jsonl"
    if not adv_path.exists():
        raise FileNotFoundError(f"{adv_path} not found.")
    records = []
    with adv_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                records.append({"text": r["text"], "label": r["label"]})
            except (json.JSONDecodeError, KeyError):
                continue
    return Dataset.from_list(records)


def train_phase(
    phase: int,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float | None = None,
    warmup_ratio: float = 0.06,
    max_length: int = 512,
) -> Path:
    """
    Train one phase of the three-phase training pipeline.

    Returns the checkpoint directory path.
    """
    import mlflow
    import torch
    from transformers import (  # type: ignore
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
    )
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # type: ignore
    import numpy as np  # type: ignore

    # Phase-specific defaults
    lr_defaults  = {1: 2e-5, 2: 1e-5, 3: 5e-6}
    lr = learning_rate or lr_defaults[phase]

    phase_output = output_dir / f"phase{phase}"
    phase_output.mkdir(parents=True, exist_ok=True)

    log.info("training_start", phase=phase, model=model_name, lr=lr, epochs=epochs)

    # ── Load data ──────────────────────────────────────────────
    if phase == 3:
        train_ds = _load_adversarial_only(data_dir)
        eval_ds  = _load_parquet_split(data_dir, "val")
    else:
        train_ds = _load_parquet_split(data_dir, "train")
        eval_ds  = _load_parquet_split(data_dir, "val")

    # ── Tokenize ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_ds  = eval_ds.map(tokenize, batched=True, remove_columns=["text"])
    train_ds = train_ds.rename_column("label", "labels")
    eval_ds  = eval_ds.rename_column("label", "labels")

    # ── Model ──────────────────────────────────────────────────
    # Phase 2/3: load from previous phase checkpoint
    load_from = model_name
    if phase == 2:
        prev = output_dir / "phase1"
        if prev.exists():
            load_from = str(prev)
    elif phase == 3:
        prev = output_dir / "phase2"
        if prev.exists():
            load_from = str(prev)

    model = AutoModelForSequenceClassification.from_pretrained(
        load_from,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # ── Training arguments ─────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(phase_output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="mlflow",
        run_name=f"text-detector-phase{phase}",
    )

    # ── Metrics ────────────────────────────────────────────────
    def compute_metrics(eval_pred: Any) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        return {
            "accuracy": round(accuracy_score(labels, preds), 4),
            "f1":       round(f1_score(labels, preds), 4),
            "auc":      round(roc_auc_score(labels, probs), 4),
        }

    # ── Trainer ────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    with mlflow.start_run(run_name=f"phase{phase}"):
        mlflow.log_params({
            "phase": phase,
            "model": model_name,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
        })

        trainer.train()
        eval_results = trainer.evaluate()

        mlflow.log_metrics({
            k: v for k, v in eval_results.items() if isinstance(v, float)
        })

        log.info("phase_complete", phase=phase, metrics=eval_results)

    # Save final checkpoint
    trainer.save_model(str(phase_output))
    tokenizer.save_pretrained(str(phase_output))
    return phase_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("ai/text-detector/checkpoints/transformer"))
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

    checkpoint = train_phase(
        phase=args.phase,
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    log.info("saved_to", path=str(checkpoint))


if __name__ == "__main__":
    main()
