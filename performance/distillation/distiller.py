"""
Step 96: Model distillation — create smaller, faster detector versions
for pre-screening.

Distillation trains a lightweight "student" model to mimic a large
"teacher" model. The student is 10–50× faster at inference while
retaining 85–95% of the teacher's accuracy.

Architecture:
  Text   teacher: DeBERTa-v3-base (86M params, ~200ms CPU) →
         student: DistilBERT (66M → further distilled to 6M), ~20ms CPU
  Image  teacher: EfficientNet-B4 (19M params) →
         student: MobileNetV3-Small (2.5M params), ~5ms CPU
  Audio  teacher: Wav2Vec2-base (94M params) →
         student: tiny-CNN on MFCCs (500K params), ~10ms CPU

Tiered analysis (Step 98) uses the student for fast pre-screening
and only invokes the teacher for content that the student flags as
suspicious (score > FAST_SCREEN_THRESHOLD).

Knowledge distillation loss:
  L = α * CE(student_logits, hard_labels)      # standard cross-entropy
    + (1-α) * T² * KL(soft_student, soft_teacher)  # soft label matching
  where T = temperature (controls softness of teacher distribution)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

TEMPERATURE   = 4.0      # distillation temperature
ALPHA         = 0.5      # weight for hard vs soft loss
FAST_SCREEN_THRESHOLD = 0.35   # below this → skip deep analysis (likely human)


# ── Student model definitions ─────────────────────────────────

def build_text_student(n_classes: int = 2) -> Any:
    """
    Lightweight text detector: 6-layer DistilBERT with a classification head.
    ~66M params → 40ms CPU inference per 512-token chunk.
    Further quantised to INT8 for ~20ms.
    """
    try:
        from transformers import DistilBertForSequenceClassification  # type: ignore
        return DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
        )
    except Exception:
        return _fallback_student(n_classes)


def build_image_student(n_classes: int = 2) -> Any:
    """
    Lightweight image detector: MobileNetV3-Small.
    2.5M params → 5ms GPU inference, 25ms CPU.
    """
    try:
        import timm  # type: ignore
        return timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True,
            num_classes=n_classes,
        )
    except Exception:
        return _fallback_student(n_classes)


def build_audio_student(n_classes: int = 2) -> Any:
    """
    Tiny CNN on MFCC features — ~500K parameters.
    Inputs: [batch, 39, T] MFCC tensors.
    ~10ms CPU per 30-second chunk.
    """
    try:
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv1d(39, 64,  kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )
    except Exception:
        return _fallback_student(n_classes)


def _fallback_student(n_classes: int) -> Any:
    try:
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, n_classes),
        )
    except Exception:
        return None


# ── Distillation trainer ──────────────────────────────────────

class DistillationTrainer:
    """
    Knowledge distillation: train a student to mimic a teacher model.

    The key insight: matching the teacher's soft probability distributions
    (e.g. GPT-like text = 0.62 AI, 0.38 human) transfers more knowledge
    than just matching hard labels (0 or 1).
    """

    def __init__(
        self,
        teacher:     Any,
        student:     Any,
        temperature: float = TEMPERATURE,
        alpha:       float = ALPHA,
        device:      str | None = None,
    ) -> None:
        self._teacher     = teacher
        self._student     = student
        self._temperature = temperature
        self._alpha       = alpha
        self._device_str  = device

    def distillation_loss(
        self,
        student_logits: Any,
        teacher_logits: Any,
        hard_labels:    Any,
    ) -> Any:
        """
        Combined hard + soft distillation loss.

        L = α * CE(student, hard_labels)
          + (1-α) * T² * KL(softmax(student/T) || softmax(teacher/T))
        """
        import torch
        import torch.nn.functional as F

        T = self._temperature

        # Hard loss: standard cross-entropy
        hard_loss = F.cross_entropy(student_logits, hard_labels)

        # Soft loss: KL divergence between soft distributions
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits   / T, dim=-1)
        soft_loss    = F.kl_div(soft_student, soft_teacher,
                                reduction="batchmean") * (T ** 2)

        return self._alpha * hard_loss + (1 - self._alpha) * soft_loss

    def train(
        self,
        train_loader:   Any,
        val_loader:     Any,
        epochs:         int   = 10,
        lr:             float = 5e-5,
        output_dir:     Path | None = None,
    ) -> dict[str, float]:
        """
        Run the full distillation training loop.
        Returns the best validation metrics achieved.
        """
        import torch
        import torch.optim as optim
        import mlflow  # type: ignore

        device = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        self._teacher.to(device).eval()
        self._student.to(device).train()

        optimizer = optim.AdamW(self._student.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best_val_acc = 0.0

        log.info("distillation_start",
                 epochs=epochs, lr=lr, T=self._temperature, alpha=self._alpha)

        with mlflow.start_run(run_name="distillation"):
            mlflow.log_params({
                "epochs": epochs, "lr": lr,
                "temperature": self._temperature, "alpha": self._alpha,
            })

            for epoch in range(epochs):
                # Training
                self._student.train()
                train_loss = 0.0
                n_batches  = 0

                for batch in train_loader:
                    inputs, labels = batch
                    inputs = {k: v.to(device) for k, v in inputs.items()} \
                             if isinstance(inputs, dict) else inputs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        teacher_out = self._teacher(**inputs) \
                                      if isinstance(inputs, dict) else self._teacher(inputs)
                        teacher_logits = (teacher_out.logits
                                         if hasattr(teacher_out, "logits")
                                         else teacher_out)

                    student_out    = self._student(**inputs) \
                                     if isinstance(inputs, dict) else self._student(inputs)
                    student_logits = (student_out.logits
                                      if hasattr(student_out, "logits")
                                      else student_out)

                    loss = self.distillation_loss(student_logits, teacher_logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._student.parameters(), 1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    n_batches  += 1

                scheduler.step()
                avg_loss = train_loss / max(n_batches, 1)

                # Validation
                val_metrics = self._evaluate(val_loader, device)
                val_acc     = val_metrics.get("accuracy", 0.0)

                mlflow.log_metrics({"train_loss": avg_loss, **val_metrics}, step=epoch)
                log.info("distillation_epoch",
                         epoch=epoch+1, loss=round(avg_loss, 4), **val_metrics)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if output_dir:
                        self._save_student(output_dir / "best_student.pt")

        if output_dir:
            self._save_student(output_dir / "final_student.pt")
            self._save_config(output_dir)

        return {"best_val_accuracy": best_val_acc}

    def _evaluate(self, loader: Any, device: str) -> dict[str, float]:
        import torch
        self._student.eval()
        correct = total = 0

        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()} \
                         if isinstance(inputs, dict) else inputs.to(device)
                labels = labels.to(device)
                out    = self._student(**inputs) \
                         if isinstance(inputs, dict) else self._student(inputs)
                logits = out.logits if hasattr(out, "logits") else out
                preds  = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += len(labels)

        return {"accuracy": round(correct / max(total, 1), 4)}

    def _save_student(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._student.state_dict(), path)
        log.info("student_saved", path=str(path))

    def _save_config(self, output_dir: Path) -> None:
        config = {
            "temperature": self._temperature,
            "alpha":       self._alpha,
            "fast_screen_threshold": FAST_SCREEN_THRESHOLD,
        }
        with (output_dir / "distillation_config.json").open("w") as f:
            json.dump(config, f, indent=2)


# ── Accuracy benchmark ────────────────────────────────────────

def benchmark_student_vs_teacher(
    student: Any,
    teacher: Any,
    test_data: list[tuple[Any, int]],
    device: str = "cpu",
) -> dict[str, float]:
    """
    Compare student and teacher accuracy on a test set.
    Returns accuracy, AUC, and inference time ratio.
    """
    import time
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score  # type: ignore

    student.eval(); teacher.eval()
    student.to(device); teacher.to(device)

    s_preds, t_preds, labels = [], [], []
    s_time = t_time = 0.0

    with torch.no_grad():
        for x, y in test_data:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                x = x.unsqueeze(0).to(device)

            t0 = time.perf_counter()
            s_out   = student(x)
            s_logits = s_out.logits if hasattr(s_out, "logits") else s_out
            s_prob   = float(F.softmax(s_logits, dim=-1)[0, 1].item())
            s_time  += time.perf_counter() - t0

            t0 = time.perf_counter()
            t_out   = teacher(x)
            t_logits = t_out.logits if hasattr(t_out, "logits") else t_out
            t_prob   = float(F.softmax(t_logits, dim=-1)[0, 1].item())
            t_time  += time.perf_counter() - t0

            s_preds.append(s_prob)
            t_preds.append(t_prob)
            labels.append(y)

    n = len(labels)
    s_acc  = sum(1 for p, y in zip(s_preds, labels) if (p >= 0.5) == bool(y)) / n
    t_acc  = sum(1 for p, y in zip(t_preds, labels) if (p >= 0.5) == bool(y)) / n
    s_auc  = roc_auc_score(labels, s_preds) if len(set(labels)) > 1 else 0.5
    t_auc  = roc_auc_score(labels, t_preds) if len(set(labels)) > 1 else 0.5

    speedup = t_time / max(s_time, 1e-9)

    return {
        "student_accuracy":   round(s_acc, 4),
        "teacher_accuracy":   round(t_acc, 4),
        "accuracy_retention": round(s_acc / max(t_acc, 1e-8), 4),
        "student_auc":        round(s_auc, 4),
        "teacher_auc":        round(t_auc, 4),
        "speedup_factor":     round(speedup, 2),
        "student_ms_per_sample": round(s_time / n * 1000, 2),
        "teacher_ms_per_sample": round(t_time / n * 1000, 2),
    }
