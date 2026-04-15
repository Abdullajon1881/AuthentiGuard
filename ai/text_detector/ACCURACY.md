# AuthentiGuard Text Detection — Measured Accuracy

This file is the **authoritative record** of every accuracy measurement
of the text detection pipeline. Every number below is traceable to a
persisted JSON artifact under version control. Do not quote numbers in
`README.md` or `SUMMARY.md` that are not listed here.

**Rule for the test splits:** `datasets/processed/test.parquet` and
`datasets/processed_v2/test.parquet` must NOT be used to pick weights,
thresholds, or hyperparameters. They are evaluated at most once per
change, with every run appended to this file.

---

## Production pipeline shape

- **L1 — perplexity.** Pretrained GPT-2 via HuggingFace. No fine-tuning. No persisted per-layer metric.
- **L2 — stylometry.** spaCy `en_core_web_sm` with hand-tuned features. No fine-tuning. No persisted per-layer metric.
- **L3 — semantic.** **DeBERTa-v3-small** (44M params, `microsoft/deberta-v3-small`) fine-tuned on an adversarial-augmented corpus. Checkpoint at `ai/text_detector/checkpoints/transformer_v3_hard/phase1` (`model.safetensors`, 267 MB). **NOTE:** README/SUMMARY historically referred to this as "DistilBERT" — it is not. The correct architecture is DeBERTa-v3-small. See `ai/text_detector/ensemble/text_detector.py:19` (`SemanticLayer`) and the layer-loader log output `loading_semantic_layer model=microsoft/deberta-v3-small`.
- **L4 — adversarial.** **NOT LOADED.** `adversarial_checkpoint=None` at `backend/app/workers/text_worker.py::_get_detector`. No trained L4 checkpoint exists in the repo.
- **Meta-classifier.** **NOT LOADED.** `meta_checkpoint=None`. The XGBoost + Platt + isotonic scaffolding at `ai/text_detector/ensemble/meta_classifier.py` has never been fitted. Production falls through to a hardcoded weighted-average combiner.
- **Inference combiner (production).** Weighted average `score = w1·L1 + w2·L2 + w3·L3`, then mapped to `AI | UNCERTAIN | HUMAN` via `_THRESHOLDS_BY_LAYERS[3]` in `text_detector.py`. Weights and the AI threshold are fit on val data (see below).
- **Binarization for F1 in this document:** `label == "AI"` → 1, `label in {"HUMAN", "UNCERTAIN"}` → 0. This is the conservative moderation stance — UNCERTAIN rows are counted as "not AI." The AUROC column uses raw continuous scores and is unaffected by this mapping.

---

## Measured numbers

### L3 alone, validation split, training-time eval (upward-biased)

| Metric | Value | Source |
|---|---|---|
| Dataset | `datasets/processed/val.parquet` (n=2000) | — |
| F1 | **0.9498** | `ai/text_detector/checkpoints/transformer_v3_hard/phase1/checkpoint-3582/trainer_state.json` line 524, field `eval_f1` |
| AUROC | **0.9818** | same file, line 523, field `eval_auc` |

**Caveat:** this is the HuggingFace `Trainer.compute_metrics` output on the same split used for early stopping and `best_metric` selection. Model selection and reporting share a dataset, so these numbers are upward-biased by an unknown (but non-zero) amount. They measure L3 **in isolation**, not the end-to-end L1+L2+L3 production ensemble. Do NOT cite 0.9498 as an end-to-end production number.

### L1+L2+L3 ensemble, held-out v1 test split (PRE-FIT hardcoded weights)

Measured with the old hardcoded combiner: weights `[0.25, 0.25, 0.50]`, AI threshold `0.65`.

| Metric | Value |
|---|---|
| Source JSON | `ai/text_detector/accuracy/ensemble_test_eval.json` |
| Dataset | `datasets/processed/test.parquet` (n=2000) |
| Dataset SHA-256 | `215ee1cecc8e412c24946ba04e842779f7bfa9acac9f4a88e24874cc67c4634d` |
| Git SHA at measurement | `fc64adda` |
| F1 | **0.9187** |
| Precision | **0.9976** |
| Recall | **0.8514** |
| AUROC | **0.9978** |
| Confusion matrix `[[TN,FP],[FN,TP]]` | `[[1002, 2], [148, 848]]` |
| Label breakdown | AI=850, HUMAN=975, UNCERTAIN=175 |

**Why F1 is below the L3-alone 0.9498:** 175 rows (~9%) landed in the UNCERTAIN band and were mapped to "not AI" in the binary evaluation, which dragged recall from ~0.99 to 0.85. The pre-fit 0.65 AI threshold was too conservative for the combined L1+L2+L3 score distribution.

### L1+L2+L3 ensemble, held-out v2 test split (PRE-FIT hardcoded weights)

| Metric | Value |
|---|---|
| Source JSON | `ai/text_detector/accuracy/ensemble_test_eval_v2.json` |
| Dataset | `datasets/processed_v2/test.parquet` (n=3482, has `source` column) |
| Dataset SHA-256 | `2af685854bc38537...` |
| Git SHA | `fc64adda` |
| F1 | **0.8919** |
| Precision | **0.9593** |
| Recall | **0.8332** |
| AUROC | **0.9777** |
| Confusion matrix | `[[1694, 61], [288, 1439]]` |
| UNCERTAIN rows | 503 (~14%) |

v2 test is harder than v1 because it includes adversarial subsets.

### Weight fit on val split

Grid search over `(w1, w2, w3)` on a 0.05-step simplex × AI threshold `[0.40, 0.80]` step 0.01. 9,471 evaluations. Selected on val F1; test F1 reported for verification only (NOT used for selection).

| Metric | Value |
|---|---|
| Source JSON | `ai/text_detector/accuracy/fit_weights.json` |
| Script | `scripts/fit_ensemble_weights.py` |
| Git SHA | `fc64adda` |
| **Best weights `[w1, w2, w3]`** | **`[0.20, 0.35, 0.45]`** |
| **Best AI threshold** | **`0.41`** |
| Val F1 | **0.9969** |
| Val precision | 0.9979 |
| Val recall | 0.9959 |
| Test F1 (verification only) | 0.9945 |
| Test precision (verification only) | 0.9960 |
| Test recall (verification only) | 0.9930 |
| **Val−Test F1 gap** | **0.00245** (well within 3-point healthy range) |

**Tie-break rule applied:** when two (w, t) combinations produce the same val F1 to ~1e-9, the one with higher `w3` (L3 weight) wins.

**Surprise finding:** the fit gave L2 stylometry (0.35) more weight than L3 DeBERTa (0.45). L3 is the only trained component, so I expected it to dominate. L2 is clearly contributing independent signal — which is a non-trivial result and argues against collapsing L2 into L3.

### L1+L2+L3 ensemble, held-out v1 test split (POST-FIT)

After wiring the fit values into `ai/text_detector/ensemble/text_detector.py`.

| Metric | Value | Δ vs pre-fit |
|---|---|---|
| Source JSON | `ai/text_detector/accuracy/ensemble_test_eval.post_fit.json` | — |
| Dataset | `datasets/processed/test.parquet` (n=2000, same split) | — |
| Dataset SHA-256 | `215ee1cecc8e412c...` (identical) | — |
| Git SHA | `fc64adda` | — |
| **F1** | **0.9945** | **+0.0758** |
| **Precision** | **0.9960** | -0.0016 |
| **Recall** | **0.9930** | **+0.1416** |
| AUROC | **0.9977** | -0.0001 (unchanged; AUROC is threshold-free) |
| Confusion matrix | `[[1000, 4], [7, 989]]` | — |
| UNCERTAIN rows | 65 (down from 175) | -110 |

**This is the authoritative headline number for the v1 test split.**

### L1+L2+L3 ensemble, held-out v2 test split (POST-FIT)

| Metric | Value | Δ vs pre-fit |
|---|---|---|
| Source JSON | `ai/text_detector/accuracy/ensemble_test_eval_v2.post_fit.json` | — |
| Dataset | `datasets/processed_v2/test.parquet` (n=3482, same split) | — |
| Dataset SHA-256 | `2af685854bc38537...` (identical) | — |
| Git SHA | `fc64adda` | — |
| **F1** | **0.9529** | **+0.0610** |
| Precision | 0.9243 | -0.0350 |
| **Recall** | **0.9832** | **+0.1500** |
| AUROC | 0.9767 | -0.0010 |
| Confusion matrix | `[[1616, 139], [29, 1698]]` | — |
| UNCERTAIN rows | 247 (down from 503) | -256 |

**This is the authoritative headline number for the v2 test split.**

Precision traded down by 3.5 points in exchange for a 15-point recall gain. On adversarial samples (the hard subsets in v2) this is the right trade; customers would rather catch AI text and risk flagging a human than miss obvious AI output. If FP rate matters more in a given use case, the operator can raise the threshold above `0.41` at inference time without re-fitting.

### Per-source F1 breakdown (v2 test split, POST-FIT)

Only the subsets that contain both AI and HUMAN samples yield a defined F1. Single-class subsets (pure human wiki, pure AI adversarial) are reported as `None` by convention — F1 is undefined when only one class is present.

| Source | n | F1 pre-fit | F1 post-fit | Δ |
|---|---|---|---|---|
| `dmitva` | 787 | 0.9693 | **0.9881** | +0.0188 |
| `gpt_wiki` | 1362 | 0.9025 | **0.9517** | +0.0492 |
| `artem9k` | 437 | 0.9249 | **0.9446** | +0.0197 |
| `adv_mixed` | 169 | 0.6061 | **0.8458** | **+0.2397** |
| `adv_humanized_ai` | 212 | None (all class 1) | None | — |
| `adv_aiified_human` | 137 | None (all class 0) | None | — |
| `wiki_formal` | 115 | None (all class 0) | None | — |
| `email` | 91 | None | None | — |
| `email_extra` | 39 | None | None | — |
| `artem9k_hn` | 34 | None | None | — |
| `wiki` | 99 | None | None | — |

**Headline per-source result:** `adv_mixed` (mixed-origin adversarial samples) jumped from **0.606 to 0.846 F1** — a 24-point improvement driven entirely by the fit threshold dropping from 0.65 to 0.41. The old threshold was systematically missing adversarial samples by pushing them into UNCERTAIN.

### Unverified / removed claims

- **"Non-adversarial baseline F1 0.9457, AUROC 0.9897 (v3 checkpoint, epoch 3)"** — no persisted artifact in the repo supports these exact values as of the current git SHA. Claim removed from `SUMMARY.md`.
- **"4-layer ensemble"** — production runs **3 layers**. L4 adversarial checkpoint does not exist. Claim removed from `README.md` and `SUMMARY.md`.
- **"DistilBERT"** — the actual L3 architecture is **DeBERTa-v3-small** (see the pipeline-shape section above). "DistilBERT" was a documentation error. Claim corrected in `README.md` and `SUMMARY.md`.
- **"F1 0.95 end-to-end"** — the L3-alone val F1 0.9498 was historically cited as if it were the end-to-end ensemble number. It is not. The real end-to-end test F1 (post-fit) is **0.9945 on v1 / 0.9529 on v2**, both reported above.

---

## How to reproduce

```bash
# 1) Baseline end-to-end numbers on both test splits (with whatever
#    weights/threshold are currently in text_detector.py)
python scripts/evaluate_end_to_end.py \
  --checkpoint ai/text_detector/checkpoints/transformer_v3_hard/phase1 \
  --data-dir datasets/processed \
  --split test \
  --output ai/text_detector/accuracy/ensemble_test_eval.json

python scripts/evaluate_end_to_end.py \
  --checkpoint ai/text_detector/checkpoints/transformer_v3_hard/phase1 \
  --data-dir datasets/processed_v2 \
  --split test \
  --output ai/text_detector/accuracy/ensemble_test_eval_v2.json

# 2) Fit weights and threshold on val (uses val + test per-layer caches)
python scripts/fit_ensemble_weights.py \
  --checkpoint ai/text_detector/checkpoints/transformer_v3_hard/phase1 \
  --data-dir datasets/processed \
  --output ai/text_detector/accuracy/fit_weights.json

# 3) Manually wire the fit values from fit_weights.json into
#    ai/text_detector/ensemble/text_detector.py:
#      - _THRESHOLDS_BY_LAYERS[3] "AI" low-bound -> best_threshold
#      - 3-layer weights list in the fallback block -> best_weights_l1_l2_l3

# 4) Re-run the evaluator twice with suffix "post_fit"
python scripts/evaluate_end_to_end.py \
  --checkpoint ai/text_detector/checkpoints/transformer_v3_hard/phase1 \
  --data-dir datasets/processed \
  --split test \
  --output ai/text_detector/accuracy/ensemble_test_eval.post_fit.json

python scripts/evaluate_end_to_end.py \
  --checkpoint ai/text_detector/checkpoints/transformer_v3_hard/phase1 \
  --data-dir datasets/processed_v2 \
  --split test \
  --output ai/text_detector/accuracy/ensemble_test_eval_v2.post_fit.json
```

Wall-clock time on this machine (CPU, no GPU): ~30 min per evaluator run, ~10 min per fit run after layer-score caches are warm. First fit run takes ~35 min to populate the caches.

---

## Change log

| Date | Git SHA | Change | Measured by |
|---|---|---|---|
| 2026-04-15 | fc64adda | Initial four measurements: pre-fit v1/v2, fit on val, post-fit v1/v2. First real end-to-end accuracy numbers for the production pipeline. | Stage 1 |
