"""
Build adversarial text detection dataset v2.

Combines multiple sources to ensure overlapping styles between human and AI:
  1. aadityaubhat/GPT-wiki-intro  — paired wiki intros (same topic, same formality)
  2. dmitva/human_ai_generated_text — paired student essays vs AI rewrites
  3. artem9k/ai-text-detection-pile — broad AI/human mix (with normalization)
  4. aeslc — real email bodies (casual human)
  5. squad — Wikipedia passages (formal human)

Output: datasets/processed_v2/{train,val,test}.parquet
Columns: ["text", "label", "source", "tone"]
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Text normalization ────────────────────────────────────────

# Patterns to strip from text
TITLE_PATTERNS = [
    r"^.*?\b(Essay|Report|Research Paper|Presentation|Assessment|Annotated Bibliography|Coursework|Dissertation|Case Study|Critical Writing|Term Paper)\b.*?\n+",
    r"^Table of Contents\n.*?\n\n",
    r"^\d+\.\s+[A-Z].*?\n",  # numbered headings
    r"^(Introduction|Conclusion|Abstract|References|Bibliography)\s*\n",
]

HEADER_PATTERNS = [
    r"^#{1,6}\s+.*\n",          # markdown headers
    r"^\*{3,}\s*\n",            # horizontal rules
    r"^-{3,}\s*\n",
    r"^\s*\d+\.\s+",            # numbered list at start
    r"^\s*[\*\-\u2022]\s+",     # bullet points at start
]


def normalize_text(text: str) -> str:
    """Strip formatting artifacts that leak authorship signals."""
    # Remove common essay/report titles
    for pat in TITLE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove markdown-style headers
    for pat in HEADER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.MULTILINE)

    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Normalize unicode quotes and dashes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    return text


def classify_tone(text: str) -> str:
    """Heuristic tone classifier: casual, formal, or medium."""
    lower = text.lower()
    casual_signals = sum([
        "lol" in lower, "lmao" in lower, "tbh" in lower, "ngl" in lower,
        "idk" in lower, "omg" in lower, "haha" in lower, "gonna" in lower,
        "wanna" in lower, "kinda" in lower, "gotta" in lower,
        "!!" in text, "..." in text, "??" in text,
        bool(re.search(r"\bi\b", lower)),  # lowercase "i"
    ])
    formal_signals = sum([
        "furthermore" in lower, "moreover" in lower, "consequently" in lower,
        "nevertheless" in lower, "subsequently" in lower, "therefore" in lower,
        "in conclusion" in lower, "as demonstrated" in lower,
        "significant" in lower, "approximately" in lower,
        bool(re.search(r"\bcite\b|\breference\b|\banalysis\b", lower)),
    ])

    if casual_signals >= 3:
        return "casual"
    elif formal_signals >= 2:
        return "formal"
    else:
        return "medium"


MIN_LEN = 80
MAX_LEN = 2000


def valid_text(text: str) -> bool:
    """Check if text is usable."""
    if not text or len(text.strip()) < MIN_LEN:
        return False
    # Skip texts that are mostly non-ASCII (broken encoding)
    ascii_frac = sum(1 for c in text if ord(c) < 128) / len(text)
    return ascii_frac > 0.85


def truncate(text: str) -> str:
    """Truncate to MAX_LEN at a sentence boundary."""
    if len(text) <= MAX_LEN:
        return text
    # Find last sentence-ending before MAX_LEN
    trunc = text[:MAX_LEN]
    last_period = max(trunc.rfind(". "), trunc.rfind(".\n"))
    if last_period > MIN_LEN:
        return trunc[:last_period + 1]
    return trunc


# ── Dataset loaders ───────────────────────────────────────────

def load_gpt_wiki_intro(cap: int) -> list[dict]:
    """Paired wiki intros — same topic, same formality, different author."""
    from datasets import load_dataset

    print(f"  [1/5] GPT-wiki-intro (cap={cap})...")
    ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train", streaming=True)

    rows = []
    for row in ds:
        if len(rows) >= cap * 2:
            break

        human = normalize_text(row.get("wiki_intro", ""))
        ai = normalize_text(row.get("generated_intro", ""))

        if valid_text(human):
            rows.append({
                "text": truncate(human),
                "label": 0,
                "source": "gpt_wiki",
                "tone": classify_tone(human),
            })
        if valid_text(ai):
            rows.append({
                "text": truncate(ai),
                "label": 1,
                "source": "gpt_wiki",
                "tone": classify_tone(ai),
            })

        if len(rows) % 5000 == 0 and len(rows) > 0:
            print(f"    {len(rows)} rows...")

    print(f"    Done: {len(rows)} rows")
    return rows


def load_dmitva(cap: int) -> list[dict]:
    """Paired student essays vs AI rewrites."""
    from datasets import load_dataset

    print(f"  [2/5] dmitva/human_ai_generated_text (cap={cap})...")
    ds = load_dataset("dmitva/human_ai_generated_text", split="train", streaming=True)

    rows = []
    for row in ds:
        if len(rows) >= cap * 2:
            break

        human = normalize_text(row.get("human_text", ""))
        ai = normalize_text(row.get("ai_text", ""))

        if valid_text(human):
            rows.append({
                "text": truncate(human),
                "label": 0,
                "source": "dmitva",
                "tone": classify_tone(human),
            })
        if valid_text(ai):
            rows.append({
                "text": truncate(ai),
                "label": 1,
                "source": "dmitva",
                "tone": classify_tone(ai),
            })

        if len(rows) % 5000 == 0 and len(rows) > 0:
            print(f"    {len(rows)} rows...")

    print(f"    Done: {len(rows)} rows")
    return rows


def load_artem9k(cap: int) -> list[dict]:
    """Broad AI/human pile — with aggressive normalization.

    Note: In this dataset, all ~1M human rows come first, then ~364k AI rows.
    We collect human and AI in separate passes to avoid streaming through 1M+ rows.
    """
    from datasets import load_dataset

    print(f"  [3/5] artem9k/ai-text-detection-pile (cap={cap})...")

    human_rows = []
    ai_rows = []

    # Pass 1: collect human rows (they appear first in the dataset)
    print("    Collecting human rows...")
    ds = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
    for row in ds:
        if len(human_rows) >= cap:
            break
        if row.get("source", "") != "human":
            continue
        text = normalize_text(row.get("text", ""))
        if not valid_text(text):
            continue
        human_rows.append({
            "text": truncate(text),
            "label": 0,
            "source": "artem9k",
            "tone": classify_tone(text),
        })
        if len(human_rows) % 2500 == 0:
            print(f"      {len(human_rows)} human...")
    print(f"    Got {len(human_rows)} human rows")

    # Pass 2: collect AI rows (start at ~1M offset, skip human rows fast)
    print("    Collecting AI rows (skipping human block)...")
    ds = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
    skipped = 0
    for row in ds:
        source_field = row.get("source", "")
        if source_field == "human":
            skipped += 1
            if skipped % 200000 == 0:
                print(f"      Skipped {skipped} human rows...")
            continue
        if len(ai_rows) >= cap:
            break
        text = normalize_text(row.get("text", ""))
        if not valid_text(text):
            continue
        ai_rows.append({
            "text": truncate(text),
            "label": 1,
            "source": "artem9k",
            "tone": classify_tone(text),
        })
        if len(ai_rows) % 2500 == 0:
            print(f"      {len(ai_rows)} AI...")
    print(f"    Got {len(ai_rows)} AI rows")

    rows = human_rows + ai_rows
    print(f"    Done: {len(rows)} rows ({len(human_rows)} human, {len(ai_rows)} AI)")
    return rows


def load_emails_as_human(cap: int) -> list[dict]:
    """Real email bodies — casual human writing."""
    from datasets import load_dataset

    print(f"  [4/5] aeslc emails (cap={cap})...")
    ds = load_dataset("aeslc", split="train", streaming=True)

    rows = []
    for row in ds:
        if len(rows) >= cap:
            break
        text = normalize_text(row.get("email_body", ""))
        if valid_text(text):
            rows.append({
                "text": truncate(text),
                "label": 0,
                "source": "email",
                "tone": "casual",
            })

    print(f"    Done: {len(rows)} rows")
    return rows


def load_wiki_as_human(cap: int) -> list[dict]:
    """Wikipedia passages from SQuAD — formal human writing."""
    from datasets import load_dataset

    print(f"  [5/5] SQuAD wiki passages (cap={cap})...")
    ds = load_dataset("squad", split="train", streaming=True)

    seen = set()
    rows = []
    for row in ds:
        if len(rows) >= cap:
            break
        text = normalize_text(row.get("context", ""))
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # SQuAD has many duplicate contexts
        if text_hash in seen:
            continue
        seen.add(text_hash)

        if valid_text(text):
            rows.append({
                "text": truncate(text),
                "label": 0,
                "source": "wiki",
                "tone": "formal",
            })

    print(f"    Done: {len(rows)} rows")
    return rows


# ── Main pipeline ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=50000,
                        help="Target total samples (will be balanced 50/50)")
    args = parser.parse_args()

    import pandas as pd

    output_dir = ROOT / "datasets" / "processed_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_per_class = args.target_size // 2
    # Allocate across sources
    # GPT-wiki: ~10k pairs (20k total) — best quality, same-format pairs
    # dmitva: ~5k pairs (10k total) — student + AI rewrites
    # artem9k: ~5k per class (10k total) — broad, normalized
    # emails: ~3k (human casual)
    # wiki: ~3k (human formal)
    # Total targets: ~25k human + ~25k AI

    print(f"=== DATASET V2 BUILD (target: {args.target_size}) ===\n")

    all_rows = []

    # Source 1: GPT-wiki-intro (paired, same topic)
    all_rows.extend(load_gpt_wiki_intro(cap=10000))

    # Source 2: dmitva (paired, student vs AI)
    all_rows.extend(load_dmitva(cap=5000))

    # Source 3: artem9k (broad, normalized)
    all_rows.extend(load_artem9k(cap=5000))

    # Source 4: Email bodies (casual human)
    all_rows.extend(load_emails_as_human(cap=3000))

    # Source 5: Wiki passages (formal human)
    all_rows.extend(load_wiki_as_human(cap=3000))

    # ── Deduplicate ────────────────────────────────────────────
    print(f"\nTotal raw rows: {len(all_rows)}")
    seen_hashes = set()
    deduped = []
    for row in all_rows:
        h = hashlib.md5(row["text"].encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(row)
    print(f"After dedup: {len(deduped)} (removed {len(all_rows) - len(deduped)})")

    # ── Balance 50/50 ──────────────────────────────────────────
    random.seed(42)
    human_rows = [r for r in deduped if r["label"] == 0]
    ai_rows = [r for r in deduped if r["label"] == 1]

    random.shuffle(human_rows)
    random.shuffle(ai_rows)

    min_count = min(len(human_rows), len(ai_rows), target_per_class)
    human_rows = human_rows[:min_count]
    ai_rows = ai_rows[:min_count]

    all_balanced = human_rows + ai_rows
    random.shuffle(all_balanced)

    df = pd.DataFrame(all_balanced)
    print(f"\nBalanced: {len(df)} samples ({min_count} per class)")

    # ── Fixed-length windowing (512–1024 chars) ─────────────────
    # Remove length as a predictive feature by forcing all samples
    # into the same character range.
    CROP_MIN = 512
    CROP_MAX = 1024

    def crop_to_range(text: str) -> str | None:
        """Randomly crop a continuous segment of 512–1024 chars.
        Returns None if text is too short."""
        if len(text) < CROP_MIN:
            return None  # will be dropped
        if len(text) <= CROP_MAX:
            return text
        # Pick a random target length within range
        target = random.randint(CROP_MIN, CROP_MAX)
        max_start = len(text) - target
        start = random.randint(0, max_start)
        # Try to start at a sentence boundary
        sentence_start = text.find(". ", start)
        if sentence_start != -1 and sentence_start < start + 100:
            start = sentence_start + 2
        segment = text[start : start + target]
        # Try to end at a sentence boundary
        last_period = segment.rfind(".")
        if last_period > CROP_MIN * 0.8:
            segment = segment[: last_period + 1]
        return segment

    print(f"\n  Applying fixed-length windowing ({CROP_MIN}-{CROP_MAX} chars)...")
    before_count = len(df)
    df["text"] = df["text"].apply(crop_to_range)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    dropped = before_count - len(df)
    print(f"  Dropped {dropped} samples shorter than {CROP_MIN} chars")

    # Deduplicate again after cropping (random crops can create collisions)
    pre_dedup = len(df)
    df["_hash"] = df["text"].apply(lambda t: hashlib.md5(t.encode()).hexdigest())
    df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"]).reset_index(drop=True)
    post_dedup = len(df)
    if pre_dedup != post_dedup:
        print(f"  Post-crop dedup: removed {pre_dedup - post_dedup}")

    # ── Tone reclassification ───────────────────────────────────
    # Reclassify tone after cropping (segments may differ from full text)
    df["tone"] = df["text"].apply(classify_tone)

    # Note: We do NOT aggressively balance tones. The main anti-artifact
    # defense is multi-source data + same-topic pairs + length normalization.
    # Tone is tracked for monitoring, not used as a hard balancing criterion,
    # because casual AI text is naturally rare and forcing balance would
    # destroy dataset size.

    # Re-balance 50/50 after cropping
    human_count = len(df[df["label"] == 0])
    ai_count = len(df[df["label"] == 1])
    min_count = min(human_count, ai_count)
    if human_count != ai_count:
        for label in [0, 1]:
            sub = df[df["label"] == label]
            if len(sub) > min_count:
                drop_idx = sub.sample(n=len(sub) - min_count, random_state=42).index
                df = df.drop(drop_idx)
        df = df.reset_index(drop=True)

    print(f"\n  Final: {len(df)} samples ({len(df[df['label']==0])} human, {len(df[df['label']==1])} AI)")

    # ── Length distribution check ──────────────────────────────
    for label in [0, 1]:
        sub = df[df["label"] == label]
        name = "Human" if label == 0 else "AI"
        print(f"  {name}: mean_len={sub['text'].str.len().mean():.0f}, "
              f"median={sub['text'].str.len().median():.0f}, "
              f"std={sub['text'].str.len().std():.0f}")

    # ── Verify length alignment ───────────────────────────────
    human_mean = df[df["label"] == 0]["text"].str.len().mean()
    ai_mean = df[df["label"] == 1]["text"].str.len().mean()
    diff_pct = abs(human_mean - ai_mean) / max(human_mean, ai_mean) * 100
    print(f"  Length difference: {diff_pct:.1f}% (target: <5%)")
    if diff_pct >= 5:
        print(f"  WARNING: Length difference {diff_pct:.1f}% exceeds 5% threshold!")

    # ── Final tone distribution ───────────────────────────────
    print(f"\nFinal tone distribution:")
    for label in [0, 1]:
        name = "Human" if label == 0 else "AI"
        sub = df[df["label"] == label]
        tone_counts = sub["tone"].value_counts().to_dict()
        total = len(sub)
        pcts = {t: f"{c/total*100:.0f}%" for t, c in tone_counts.items()}
        print(f"  {name}: {tone_counts} ({pcts})")

    # ── Source distribution ────────────────────────────────────
    print(f"\nSource distribution:")
    for label in [0, 1]:
        name = "Human" if label == 0 else "AI"
        sub = df[df["label"] == label]
        src_counts = sub["source"].value_counts().to_dict()
        print(f"  {name}: {src_counts}")

    # ── Shuffle and split 80/10/10 ─────────────────────────────
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # ── Verify no cross-split duplicates ───────────────────────
    train_h = set(hashlib.md5(t.encode()).hexdigest() for t in train_df["text"])
    val_h = set(hashlib.md5(t.encode()).hexdigest() for t in val_df["text"])
    test_h = set(hashlib.md5(t.encode()).hexdigest() for t in test_df["text"])

    overlap_tv = len(train_h & val_h)
    overlap_tt = len(train_h & test_h)
    overlap_vt = len(val_h & test_h)
    if overlap_tv + overlap_tt + overlap_vt > 0:
        print(f"\n  Cross-split overlaps: train-val={overlap_tv}, train-test={overlap_tt}, val-test={overlap_vt}")
        print(f"  Removing overlapping samples from val/test...")
        val_df = val_df[~val_df["text"].apply(lambda t: hashlib.md5(t.encode()).hexdigest()).isin(train_h)]
        test_df = test_df[~test_df["text"].apply(lambda t: hashlib.md5(t.encode()).hexdigest()).isin(train_h | val_h)]
    print(f"\nCross-split duplicates: CLEAN")

    # ── Save ───────────────────────────────────────────────────
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"\nSaved to {output_dir}/")
    print(f"  train: {len(train_df)}")
    print(f"  val:   {len(val_df)}")
    print(f"  test:  {len(test_df)}")

    # ── Validation: 10 random samples per class ───────────────
    print(f"\n{'='*80}")
    print("VALIDATION: 10 random HUMAN samples")
    print("=" * 80)
    for _, row in train_df[train_df["label"] == 0].sample(10, random_state=99).iterrows():
        preview = row['text'][:100].encode('ascii', 'replace').decode()
        print(f"  [{row['source']:<10s}] [{row['tone']:<7s}] {preview}")

    print(f"\n{'='*80}")
    print("VALIDATION: 10 random AI samples")
    print("=" * 80)
    for _, row in train_df[train_df["label"] == 1].sample(10, random_state=99).iterrows():
        preview = row['text'][:100].encode('ascii', 'replace').decode()
        print(f"  [{row['source']:<10s}] [{row['tone']:<7s}] {preview}")


if __name__ == "__main__":
    main()
