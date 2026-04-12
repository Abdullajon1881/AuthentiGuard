"""
Phase 1B: Calibrate Layer 1 perplexity constants using sample data.

Loads calibration_samples.json, runs GPT-2 perplexity on each sample,
computes distribution statistics, and recommends updated constants for
ai/text_detector/layers/layer1_perplexity.py.

Usage:
    python scripts/calibrate_perplexity.py

Requires: torch, transformers (installed via requirements/ml.txt)
Runtime: ~5-8 minutes on CPU (GPT-2 download on first run adds 5-10 min)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if len(s.split()) >= 4]


def compute_perplexity(model, tokenizer, device, sentence: str) -> float | None:
    import torch

    tokens = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    input_ids = tokens["input_ids"].to(device)
    if input_ids.shape[1] < 3:
        return None
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    return math.exp(loss)


def main():
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    samples_path = ROOT / "scripts" / "calibration_samples.json"
    if not samples_path.exists():
        print(f"ERROR: {samples_path} not found")
        sys.exit(1)

    with open(samples_path) as f:
        samples = json.load(f)

    human_samples = [s for s in samples if s["label"] == "human"]
    ai_samples = [s for s in samples if s["label"] == "ai"]
    print(f"Loaded {len(human_samples)} human + {len(ai_samples)} AI samples")

    # Load GPT-2
    print("Loading GPT-2 (downloads ~500MB on first run)...")
    t0 = time.time()
    device = torch.device("cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(device)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Compute per-sample statistics
    human_stats = []
    ai_stats = []

    for group, samples_list, label in [
        (human_stats, human_samples, "human"),
        (ai_stats, ai_samples, "ai"),
    ]:
        for i, sample in enumerate(samples_list):
            sentences = _split_sentences(sample["text"])
            if not sentences:
                sentences = [sample["text"]]

            ppls = []
            for sent in sentences:
                ppl = compute_perplexity(model, tokenizer, device, sent)
                if ppl is not None:
                    ppls.append(ppl)

            if not ppls:
                continue

            mean_ppl = sum(ppls) / len(ppls)
            if len(ppls) > 1:
                variance = sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)
                std_ppl = math.sqrt(variance)
            else:
                std_ppl = 0.0

            low_ppl_frac = sum(1 for p in ppls if p < 50.0) / len(ppls)

            group.append({
                "mean_ppl": mean_ppl,
                "std_ppl": std_ppl,
                "burstiness": std_ppl,
                "low_ppl_frac": low_ppl_frac,
                "n_sentences": len(ppls),
                "all_ppls": ppls,
            })

            if (i + 1) % 5 == 0:
                print(f"  [{label}] {i + 1}/{len(samples_list)} processed")

    # Aggregate statistics
    def stats(values: list[float]) -> dict:
        if not values:
            return {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0, "p25": 0, "p75": 0}
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / n) if n > 1 else 0
        sorted_v = sorted(values)
        median = sorted_v[n // 2]
        p25 = sorted_v[n // 4]
        p75 = sorted_v[3 * n // 4]
        return {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "median": round(median, 2),
            "min": round(sorted_v[0], 2),
            "max": round(sorted_v[-1], 2),
            "p25": round(p25, 2),
            "p75": round(p75, 2),
        }

    human_ppls = [s["mean_ppl"] for s in human_stats]
    ai_ppls = [s["mean_ppl"] for s in ai_stats]
    human_burst = [s["burstiness"] for s in human_stats]
    ai_burst = [s["burstiness"] for s in ai_stats]
    human_low_frac = [s["low_ppl_frac"] for s in human_stats]
    ai_low_frac = [s["low_ppl_frac"] for s in ai_stats]

    h_ppl_stats = stats(human_ppls)
    a_ppl_stats = stats(ai_ppls)
    h_burst_stats = stats(human_burst)
    a_burst_stats = stats(ai_burst)
    h_low_stats = stats(human_low_frac)
    a_low_stats = stats(ai_low_frac)

    # Print results
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    print(f"\nSamples processed: {len(human_stats)} human, {len(ai_stats)} AI")

    print("\n--- Mean Perplexity ---")
    print(f"  Human: mean={h_ppl_stats['mean']}, std={h_ppl_stats['std']}, "
          f"median={h_ppl_stats['median']}, range=[{h_ppl_stats['min']}, {h_ppl_stats['max']}]")
    print(f"  AI:    mean={a_ppl_stats['mean']}, std={a_ppl_stats['std']}, "
          f"median={a_ppl_stats['median']}, range=[{a_ppl_stats['min']}, {a_ppl_stats['max']}]")

    print("\n--- Burstiness (std of per-sentence ppl) ---")
    print(f"  Human: mean={h_burst_stats['mean']}, std={h_burst_stats['std']}, "
          f"median={h_burst_stats['median']}")
    print(f"  AI:    mean={a_burst_stats['mean']}, std={a_burst_stats['std']}, "
          f"median={a_burst_stats['median']}")

    print("\n--- Low Perplexity Fraction (sentences < threshold) ---")
    print(f"  Human: mean={h_low_stats['mean']}, std={h_low_stats['std']}")
    print(f"  AI:    mean={a_low_stats['mean']}, std={a_low_stats['std']}")

    # ASCII histogram
    print("\n--- Perplexity Distribution ---")
    all_ppls = human_ppls + ai_ppls
    if all_ppls:
        bin_edges = [0, 20, 40, 60, 80, 100, 120, 150, 200, 300, 500]
        print(f"  {'Bin':>10}  {'Human':>6}  {'AI':>6}")
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            h_count = sum(1 for p in human_ppls if lo <= p < hi)
            a_count = sum(1 for p in ai_ppls if lo <= p < hi)
            h_bar = "H" * h_count
            a_bar = "A" * a_count
            print(f"  {lo:>4}-{hi:<4}  {h_count:>6}  {a_count:>6}  {h_bar}{a_bar}")

    # Compute separation quality
    if human_ppls and ai_ppls:
        # Fisher's discriminant ratio
        h_mean, a_mean = h_ppl_stats["mean"], a_ppl_stats["mean"]
        h_std, a_std = max(h_ppl_stats["std"], 0.1), max(a_ppl_stats["std"], 0.1)
        fisher = abs(h_mean - a_mean) / math.sqrt(h_std**2 + a_std**2)
        print(f"\n  Fisher discriminant ratio: {fisher:.3f} (>1.0 = good separation)")

    # Recommend constants
    print("\n" + "=" * 70)
    print("RECOMMENDED CONSTANTS")
    print("=" * 70)

    rec_human_ppl_mean = round(h_ppl_stats["mean"], 1)
    rec_human_ppl_std = round(h_ppl_stats["std"], 1)
    rec_ai_ppl_mean = round(a_ppl_stats["mean"], 1)
    rec_ai_ppl_std = round(a_ppl_stats["std"], 1)

    # LOW_PPL_THRESHOLD: midpoint between AI mean and AI p75
    rec_low_ppl = round((a_ppl_stats["mean"] + a_ppl_stats["p75"]) / 2, 1)

    # Burstiness threshold: midpoint between AI and human burstiness means
    rec_burst_thresh = round((a_burst_stats["mean"] + h_burst_stats["mean"]) / 2, 1)

    print(f"\n  HUMAN_PPL_MEAN = {rec_human_ppl_mean}  (was 120.0)")
    print(f"  HUMAN_PPL_STD  = {rec_human_ppl_std}  (was 45.0)")
    print(f"  AI_PPL_MEAN    = {rec_ai_ppl_mean}  (was 35.0)")
    print(f"  AI_PPL_STD     = {rec_ai_ppl_std}  (was 18.0)")
    print(f"  LOW_PPL_THRESHOLD      = {rec_low_ppl}  (was 50.0)")
    print(f"  LOW_BURSTINESS_THRESHOLD = {rec_burst_thresh}  (was 20.0)")

    # Signal weight recommendation
    print("\n--- Signal Weight Analysis ---")
    if human_ppls and ai_ppls:
        # Test current scoring on all samples
        current_correct = 0
        recommended_correct = 0
        total = 0

        ppl_range_cur = 120.0 - 35.0
        ppl_range_rec = rec_human_ppl_mean - rec_ai_ppl_mean

        for group_stats, true_label in [(human_stats, 0), (ai_stats, 1)]:
            for s in group_stats:
                total += 1
                mean_ppl = s["mean_ppl"]
                burstiness = s["burstiness"]
                low_ppl_frac = s["low_ppl_frac"]

                # Current formula
                ppl_sig = max(0, min(1, (120.0 - mean_ppl) / ppl_range_cur))
                burst_sig = 1.0 - min(burstiness / (45.0 * 2), 1.0)
                raw = 0.60 * ppl_sig + 0.30 * burst_sig + 0.10 * low_ppl_frac
                cur_score = 1.0 / (1.0 + math.exp(-(raw - 0.5) * 6))
                cur_pred = 1 if cur_score > 0.5 else 0
                if cur_pred == true_label:
                    current_correct += 1

                # Recommended formula
                ppl_sig_r = max(0, min(1, (rec_human_ppl_mean - mean_ppl) / max(ppl_range_rec, 1)))
                burst_sig_r = 1.0 - min(burstiness / (rec_human_ppl_std * 2), 1.0)
                raw_r = 0.60 * ppl_sig_r + 0.30 * burst_sig_r + 0.10 * low_ppl_frac
                rec_score = 1.0 / (1.0 + math.exp(-(raw_r - 0.5) * 6))
                rec_pred = 1 if rec_score > 0.5 else 0
                if rec_pred == true_label:
                    recommended_correct += 1

        cur_acc = current_correct / total if total else 0
        rec_acc = recommended_correct / total if total else 0
        print(f"  Current constants accuracy:     {current_correct}/{total} = {cur_acc:.1%}")
        print(f"  Recommended constants accuracy:  {recommended_correct}/{total} = {rec_acc:.1%}")
        print(f"  Improvement: {rec_acc - cur_acc:+.1%}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("\nDone. Update constants in ai/text_detector/layers/layer1_perplexity.py")


if __name__ == "__main__":
    main()
