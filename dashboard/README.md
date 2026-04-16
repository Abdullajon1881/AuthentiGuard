# AuthentiGuard Control Tower Dashboard

Operator-focused confidence calibration dashboard for the text detection system.

## Setup

```bash
pip install -r dashboard/requirements.txt
```

## Generate demo data (first time only)

```bash
python dashboard/generate_demo_data.py
```

This creates 7 days of realistic prediction logs + daily metrics + drift data
so the dashboard has content on first launch.

## Launch

```bash
cd authentiguard
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

## What it shows

| Panel | Description |
|---|---|
| **System Trust State** | GREEN / YELLOW / RED banner based on coverage, reliability, and drift |
| **KPI row** | Total predictions, coverage %, eval reliability, PSI drift score |
| **Live stream** | Latest 100 predictions with score, label, margin, zone, version |
| **Label distribution** | AI / HUMAN / UNCERTAIN pie chart |
| **Confidence histogram** | Score distribution with zone boundaries marked |
| **Calibration curve** | Predicted probability vs actual positive rate + ECE + Brier |
| **Drift indicator** | PSI per day with STABLE / MODERATE / SIGNIFICANT coloring |
| **Coverage over time** | Coverage + abstain rate trend with GREEN threshold line |
| **Confidence margin** | Distribution of (score - 0.41) showing how far predictions are from the legacy threshold |
| **Model version comparison** | F1 / precision / recall / AUROC / reliability / coverage across Stage 1, Stage 2, and reliability-gated evaluations |

## Trust state logic

| State | Criteria |
|---|---|
| GREEN | coverage >= 60% AND reliability >= 90% AND PSI < 0.10 |
| YELLOW | coverage >= 40% AND reliability >= 70% AND PSI < 0.25 |
| RED | anything worse |

## Data sources (read-only)

- `logs/predictions/*.jsonl` — prediction logs from `backend/app/observability/prediction_log.py`
- `metrics/daily_metrics.json` — from `scripts/compute_daily_metrics.py`
- `metrics/drift.json` — from `scripts/compute_drift.py`
- `ai/text_detector/accuracy/*.json` — evaluation artifacts from Stages 1-2 + reliability layer

No model changes. No retraining. Plug-and-play.
