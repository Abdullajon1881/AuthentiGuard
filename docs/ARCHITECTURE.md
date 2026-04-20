# AuthentiGuard Text Detection — Architecture

Single entry point, three layers, one decision policy.

```
            text ─┐
                  ▼
         pipeline.analyze()           ai/text_detector/pipeline.py
                  │  singleton
                  ▼
              TextDetector            ai/text_detector/detector.py
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
  L1 GPT-2    L2 stylometry  L3 DeBERTa-v3-small     ai/text_detector/layers.py
 perplexity    lexical/POS    (fine-tuned checkpoint)
    │             │             │
    └──────┬──────┴──────┬──────┘
           ▼             ▼
  Stage 2: LR + isotonic calibrator   ai/text_detector/checkpoints/meta_*.joblib
           │        (fallback: fixed weights [0.20, 0.35, 0.45])
           ▼
  Reliability-gated decision          AI ≥ 0.70 · HUMAN ≤ 0.30 · else UNCERTAIN
           │        G1: <50 words → UNCERTAIN
           ▼
       EnsembleResult                 score · label · confidence · evidence
```

## File layout

```
ai/text_detector/
    pipeline.py       single entry point — analyze(text) -> EnsembleResult
    detector.py       TextDetector orchestration + reliability gate
    layers.py         L1 PerplexityLayer, L2 StylometryLayer, L3 SemanticLayer
    meta.py           EnsembleResult, build_feature_vector, MetaClassifier (legacy)
    checkpoints/
        transformer_v3_hard/phase1/      DeBERTa-v3-small fine-tuned weights
        meta_classifier.joblib           Stage 2 LR
        meta_calibrator.joblib           isotonic calibrator bundle
```

Model version string: `MODEL_VERSION = "3.2-g2-removed-product-output"` (in `detector.py`).

## Import contract

All consumers call through `pipeline.py`:

```python
from ai.text_detector.pipeline import analyze, MODEL_VERSION
result = analyze("some text...")
```

Production consumer: `backend/app/workers/text_worker.py`.

## Sanity check

```bash
python scripts/sanity_check.py
```

Verifies end-to-end detection on two fixtures in <10s after models are cached.

## Archived (not used at inference)

`archive/text_detector/` holds the previous package layout, training code,
evaluation scripts, legacy L4 adversarial layer, and the XGBoost meta scaffold.
Kept for traceability; not loaded by the runtime.
