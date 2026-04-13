"""
Hard example mining + threshold calibration.

1. Run model on training set, find all misclassified samples
2. Save hard_examples.json for oversampled retraining
3. Sweep thresholds on validation set to find optimal F1
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,
)


def load_model(checkpoint_path: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


def predict_all(tokenizer, model, texts: list[str], batch_size: int = 32):
    """Predict probabilities for all texts."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy().tolist())
        if (i // batch_size) % 50 == 0:
            print(f"  Predicted {i + len(batch)}/{len(texts)}...")
    return np.array(all_probs)


def mine_hard_examples(checkpoint: str, data_dir: str, output_path: str):
    """Find all misclassified training samples and save to JSON."""
    import pandas as pd

    tokenizer, model = load_model(checkpoint)

    train_df = pd.read_parquet(Path(data_dir) / "train.parquet")
    texts = train_df["text"].tolist()
    labels = train_df["label"].values
    sources = train_df["source"].values if "source" in train_df.columns else ["unknown"] * len(texts)

    print(f"\n{'='*80}")
    print(f"MINING HARD EXAMPLES from training set ({len(texts)} samples)")
    print(f"{'='*80}")

    probs = predict_all(tokenizer, model, texts)
    preds = (probs > 0.5).astype(int)

    # Find false positives and false negatives
    fp_mask = (preds == 1) & (labels == 0)  # predicted AI, actually human
    fn_mask = (preds == 0) & (labels == 1)  # predicted human, actually AI

    false_positives = []
    false_negatives = []

    for idx in np.where(fp_mask)[0]:
        false_positives.append({
            "text": texts[idx],
            "label": int(labels[idx]),
            "predicted_prob": float(probs[idx]),
            "source": str(sources[idx]),
            "error_type": "false_positive",
        })

    for idx in np.where(fn_mask)[0]:
        false_negatives.append({
            "text": texts[idx],
            "label": int(labels[idx]),
            "predicted_prob": float(probs[idx]),
            "source": str(sources[idx]),
            "error_type": "false_negative",
        })

    # Sort by confidence of wrong prediction (worst errors first)
    false_positives.sort(key=lambda x: x["predicted_prob"], reverse=True)
    false_negatives.sort(key=lambda x: x["predicted_prob"])

    hard_data = {
        "checkpoint": checkpoint,
        "train_samples": len(texts),
        "train_accuracy": float(accuracy_score(labels, preds)),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hard_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Train accuracy: {hard_data['train_accuracy']:.4f}")
    print(f"  False positives (human → AI): {len(false_positives)}")
    print(f"  False negatives (AI → human): {len(false_negatives)}")
    print(f"  Total hard examples: {len(false_positives) + len(false_negatives)}")

    # Breakdown by source
    print(f"\n  Hard examples by source:")
    from collections import Counter
    fp_sources = Counter(x["source"] for x in false_positives)
    fn_sources = Counter(x["source"] for x in false_negatives)
    all_sources = set(list(fp_sources.keys()) + list(fn_sources.keys()))
    for src in sorted(all_sources):
        fp_n = fp_sources.get(src, 0)
        fn_n = fn_sources.get(src, 0)
        print(f"    {src:<25s}: {fp_n} FP, {fn_n} FN")

    # Show worst errors
    print(f"\n  Top 5 worst false positives (human predicted as AI):")
    for ex in false_positives[:5]:
        preview = ex["text"][:80].encode("ascii", "replace").decode()
        print(f"    [prob={ex['predicted_prob']:.4f}] [{ex['source']}] {preview}")

    print(f"\n  Top 5 worst false negatives (AI predicted as human):")
    for ex in false_negatives[:5]:
        preview = ex["text"][:80].encode("ascii", "replace").decode()
        print(f"    [prob={ex['predicted_prob']:.4f}] [{ex['source']}] {preview}")

    print(f"\n  Saved to {output_path}")
    return hard_data


def calibrate_threshold(checkpoint: str, data_dir: str):
    """Sweep thresholds on validation set to find optimal F1."""
    import pandas as pd

    tokenizer, model = load_model(checkpoint)

    val_df = pd.read_parquet(Path(data_dir) / "val.parquet")
    texts = val_df["text"].tolist()
    labels = val_df["label"].values
    sources = val_df["source"].values if "source" in val_df.columns else None

    print(f"\n{'='*80}")
    print(f"THRESHOLD CALIBRATION on validation set ({len(texts)} samples)")
    print(f"{'='*80}")

    probs = predict_all(tokenizer, model, texts)

    # Sweep thresholds
    thresholds = np.arange(0.30, 0.81, 0.02)
    results = []

    print(f"\n  {'Thresh':<8s} {'F1':<8s} {'Prec':<8s} {'Rec':<8s} {'Acc':<8s} {'FP':<6s} {'FN':<6s}")
    print(f"  {'-'*52}")

    best_f1 = 0
    best_threshold = 0.5

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        results.append({
            "threshold": float(thresh), "f1": f1, "precision": prec,
            "recall": rec, "accuracy": acc, "fp": fp, "fn": fn,
        })

        marker = " <<<" if f1 > best_f1 else ""
        print(f"  {thresh:<8.2f} {f1:<8.4f} {prec:<8.4f} {rec:<8.4f} {acc:<8.4f} {fp:<6d} {fn:<6d}{marker}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thresh)

    print(f"\n  OPTIMAL THRESHOLD: {best_threshold:.2f} (F1={best_f1:.4f})")

    # Check humanized AI recall specifically at various thresholds
    if sources is not None:
        adv_mask = np.isin(sources, ["adv_humanized_ai", "adv_mixed"])
        if adv_mask.any():
            print(f"\n  Adversarial sample recall at key thresholds:")
            adv_labels = labels[adv_mask]
            adv_probs = probs[adv_mask]
            adv_ai_mask = adv_labels == 1
            if adv_ai_mask.any():
                for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, best_threshold]:
                    adv_preds = (adv_probs[adv_ai_mask] >= thresh).astype(int)
                    adv_recall = adv_preds.mean()
                    print(f"    thresh={thresh:.2f}: adversarial AI recall={adv_recall:.4f} "
                          f"({adv_preds.sum()}/{len(adv_preds)})")

    # Also check per-source at optimal threshold
    if sources is not None:
        print(f"\n  Per-source accuracy at optimal threshold ({best_threshold:.2f}):")
        preds_opt = (probs >= best_threshold).astype(int)
        for src in sorted(set(sources)):
            mask = sources == src
            src_acc = accuracy_score(labels[mask], preds_opt[mask])
            n = mask.sum()
            print(f"    {src:<25s}: {src_acc:.4f} (n={n})")

    return best_threshold, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="datasets/processed_v2")
    parser.add_argument("--output", type=str, default="datasets/processed_v2/hard_examples.json")
    parser.add_argument("--skip-mining", action="store_true",
                        help="Skip mining, only calibrate threshold")
    args = parser.parse_args()

    if not args.skip_mining:
        mine_hard_examples(args.checkpoint, args.data_dir, args.output)

    optimal_threshold, _ = calibrate_threshold(args.checkpoint, args.data_dir)

    print(f"\n{'='*80}")
    print(f"NEXT STEPS")
    print(f"{'='*80}")
    print(f"  1. Retrain with hard examples:")
    print(f"     python -m ai.text_detector.training.train_transformer \\")
    print(f"       --phase 1 --data-dir datasets/processed_v2 \\")
    print(f"       --output-dir ai/text_detector/checkpoints/transformer_v3 \\")
    print(f"       --hard-examples {args.output} --oversample 3")
    print(f"  2. Use optimal threshold {optimal_threshold:.2f} instead of 0.5")


if __name__ == "__main__":
    main()
