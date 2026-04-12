"""
Prepare text detection training data from HuggingFace datasets.

Uses artem9k/ai-text-detection-pile (streaming to avoid disk cache).
Converts to balanced parquet splits for train_transformer.py.

Output: datasets/processed/{train,val,test}.parquet
Columns: ["text", "label"] where label is 0 (human) or 1 (ai)

Usage:
    python scripts/prepare_text_data.py
    python scripts/prepare_text_data.py --samples-per-class 10000
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=10000,
                        help="Max samples per class (human/ai)")
    args = parser.parse_args()

    import pandas as pd
    from datasets import load_dataset

    output_dir = ROOT / "datasets" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = args.samples_per_class
    human_rows: list[dict] = []
    ai_rows: list[dict] = []

    # ── artem9k/ai-text-detection-pile (streaming) ─────────────
    print(f"Loading ai-text-detection-pile (streaming, {cap} per class)...")
    ds = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)

    for row in ds:
        text = row.get("text", "").strip()
        source = row.get("source", "")

        # Skip short texts
        if len(text) < 100:
            continue

        # Truncate very long texts to ~2000 chars for training efficiency
        if len(text) > 2000:
            text = text[:2000]

        if source == "human" and len(human_rows) < cap:
            human_rows.append({"text": text, "label": 0})
        elif source != "human" and len(ai_rows) < cap:
            ai_rows.append({"text": text, "label": 1})

        if len(human_rows) >= cap and len(ai_rows) >= cap:
            break

        total = len(human_rows) + len(ai_rows)
        if total % 5000 == 0 and total > 0:
            print(f"  {len(human_rows)} human, {len(ai_rows)} AI...")

    print(f"  Final: {len(human_rows)} human, {len(ai_rows)} AI")

    if not human_rows or not ai_rows:
        print("ERROR: Insufficient data")
        sys.exit(1)

    # ── Balance ────────────────────────────────────────────────
    min_count = min(len(human_rows), len(ai_rows))
    random.seed(42)
    random.shuffle(human_rows)
    random.shuffle(ai_rows)
    human_rows = human_rows[:min_count]
    ai_rows = ai_rows[:min_count]

    all_rows = human_rows + ai_rows
    random.shuffle(all_rows)

    df = pd.DataFrame(all_rows)
    print(f"Balanced: {len(df)} samples ({min_count} per class)")

    # ── Split 80/10/10 ─────────────────────────────────────────
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # ── Save ───────────────────────────────────────────────────
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"\nSaved to {output_dir}/")
    print(f"  train: {len(train_df)}")
    print(f"  val:   {len(val_df)}")
    print(f"  test:  {len(test_df)}")


if __name__ == "__main__":
    main()
