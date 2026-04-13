"""
Weighted transformer training for AuthentiGuard text detection.

Supports sample-level loss weighting via a `weight` column in the dataset.
Adversarial samples get higher weight to force the model to learn harder patterns.

Usage:
    python -m ai.text_detector.training.train_transformer \
        --phase 1 \
        --data-dir datasets/processed_v2 \
        --output-dir ai/text_detector/checkpoints/transformer_v2 \
        --model distilbert-base-uncased \
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

LABEL2ID = {"human": 0, "ai": 1}
ID2LABEL = {0: "human", 1: "ai"}


def _load_parquet_split(data_dir: Path, split: str, keep_weights: bool = False) -> Any:
    """Load a preprocessed parquet split into a HuggingFace Dataset."""
    import pandas as pd
    from datasets import Dataset

    path = data_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the data pipeline first.")

    df = pd.read_parquet(path)
    cols = ["text", "label"]
    if keep_weights and "weight" in df.columns:
        cols.append("weight")
    else:
        # Default weight = 1.0
        df["weight"] = 1.0
        cols.append("weight")

    return Dataset.from_pandas(df[cols])


def _load_with_hard_examples(
    data_dir: Path, hard_examples_path: Path | None = None, oversample: int = 3,
) -> Any:
    """Load training data, optionally oversampling hard examples."""
    import pandas as pd
    from datasets import Dataset

    train_path = data_dir / "train.parquet"
    df = pd.read_parquet(train_path)

    if "weight" not in df.columns:
        df["weight"] = 1.0

    if hard_examples_path and hard_examples_path.exists():
        with open(hard_examples_path) as f:
            hard_data = json.load(f)

        hard_texts = set(h["text"] for h in hard_data["false_positives"] + hard_data["false_negatives"])
        n_hard = len(hard_texts)

        # Oversample hard examples by duplicating them
        hard_mask = df["text"].isin(hard_texts)
        hard_df = df[hard_mask]
        log.info("hard_examples_found", count=len(hard_df), target=n_hard)

        # Duplicate hard examples (oversample - 1 copies since original is already in df)
        for _ in range(oversample - 1):
            df = pd.concat([df, hard_df], ignore_index=True)

        # Boost weight for hard examples
        df.loc[df["text"].isin(hard_texts), "weight"] = df.loc[
            df["text"].isin(hard_texts), "weight"
        ].clip(lower=3.0)

        log.info("hard_oversampling", added=len(hard_df) * (oversample - 1),
                 total=len(df))

    return Dataset.from_pandas(df[["text", "label", "weight"]])


def train_phase(
    phase: int,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float | None = None,
    warmup_ratio: float = 0.1,
    max_length: int = 512,
    hard_examples_path: Path | None = None,
    oversample: int = 3,
) -> Path:
    """
    Train one phase with sample-weighted loss.
    Returns the checkpoint directory path.
    """
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
    )
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import numpy as np

    # Phase-specific defaults
    lr_defaults = {1: 2e-5, 2: 1e-5, 3: 5e-6}
    lr = learning_rate or lr_defaults[phase]

    phase_output = output_dir / f"phase{phase}"
    phase_output.mkdir(parents=True, exist_ok=True)

    log.info("training_start", phase=phase, model=model_name, lr=lr, epochs=epochs,
             weighted=True, hard_examples=str(hard_examples_path))

    # ── Load data ──────────────────────────────────────────────
    if hard_examples_path and hard_examples_path.exists():
        train_ds = _load_with_hard_examples(data_dir, hard_examples_path, oversample)
    else:
        train_ds = _load_parquet_split(data_dir, "train", keep_weights=True)
    eval_ds = _load_parquet_split(data_dir, "val", keep_weights=False)

    # Extract weights before tokenization
    train_weights = torch.tensor(train_ds["weight"], dtype=torch.float32)
    log.info("weight_stats",
             mean=float(train_weights.mean()),
             min=float(train_weights.min()),
             max=float(train_weights.max()),
             n_weighted=int((train_weights > 1.0).sum()))

    # ── Tokenize ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    # Keep weight column through tokenization
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])
    train_ds = train_ds.rename_column("label", "labels")
    eval_ds = eval_ds.rename_column("label", "labels")

    # ── Model ──────────────────────────────────────────────────
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
    total_steps = int((len(train_ds) / batch_size) * epochs)
    training_args = TrainingArguments(
        output_dir=str(phase_output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_steps=int(total_steps * warmup_ratio),
        weight_decay=0.01,
        max_grad_norm=1.0,
        # NOTE: label_smoothing handled in custom loss, not here
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
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

    # ── Custom weighted Trainer ────────────────────────────────
    class WeightedTrainer(Trainer):
        """Trainer with per-sample loss weighting + label smoothing."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            weights = inputs.pop("weight", None)

            outputs = model(**inputs)
            logits = outputs.logits

            # Cross-entropy with label smoothing
            smoothing = 0.1
            n_classes = logits.size(-1)
            log_probs = nn.functional.log_softmax(logits, dim=-1)

            # Create smoothed targets
            with torch.no_grad():
                targets = torch.zeros_like(log_probs)
                targets.fill_(smoothing / (n_classes - 1))
                targets.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)

            loss_per_sample = -(targets * log_probs).sum(dim=-1)

            # Apply sample weights
            if weights is not None:
                weights = weights.to(loss_per_sample.device)
                loss = (loss_per_sample * weights).mean()
            else:
                loss = loss_per_sample.mean()

            return (loss, outputs) if return_outputs else loss

    # ── Train ─────────────────────────────────────────────────
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    eval_results = trainer.evaluate()
    log.info("phase_complete", phase=phase, metrics=eval_results)

    # Save final checkpoint
    trainer.save_model(str(phase_output))
    tokenizer.save_pretrained(str(phase_output))
    return phase_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/processed_v2"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("ai/text_detector/checkpoints/transformer_v2"))
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hard-examples", type=Path, default=None,
                        help="Path to hard_examples.json for oversampling")
    parser.add_argument("--oversample", type=int, default=3,
                        help="Oversample factor for hard examples")
    args = parser.parse_args()

    checkpoint = train_phase(
        phase=args.phase,
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hard_examples_path=args.hard_examples,
        oversample=args.oversample,
    )
    log.info("saved_to", path=str(checkpoint))


if __name__ == "__main__":
    main()
