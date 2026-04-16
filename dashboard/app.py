"""
AuthentiGuard Confidence Calibration Dashboard — Control Tower.

A single-page Streamlit app for operators to monitor the health and
reliability of the text detection system. Reads directly from existing
JSONL prediction logs, daily-metrics JSON, drift JSON, and evaluation
artifacts — no model changes, no retraining, plug-and-play.

Launch:
    cd authentiguard
    streamlit run dashboard/app.py

Data sources (all read-only):
    logs/predictions/*.jsonl                  → live prediction stream
    metrics/daily_metrics.json                → daily aggregates
    metrics/drift.json                        → PSI per day
    ai/text_detector/accuracy/*.json          → evaluation artifacts
    ai/text_detector/accuracy/calibration_metrics.json → ECE/Brier
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths (relative to repo root) ───────────────────────────────────

_REPO = Path(__file__).resolve().parent.parent
_LOG_DIR = _REPO / "logs" / "predictions"
_METRICS_DIR = _REPO / "metrics"
_ACCURACY_DIR = _REPO / "ai" / "text_detector" / "accuracy"


# ── Data loading ─────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


@st.cache_data(ttl=30)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_available_dates() -> list[str]:
    dates: list[str] = []
    if _LOG_DIR.exists():
        for f in sorted(_LOG_DIR.glob("*.jsonl")):
            if ".samples" not in f.name:
                dates.append(f.stem)
    return dates


def load_all_predictions(dates: list[str]) -> list[dict]:
    all_rows: list[dict] = []
    for d in dates:
        all_rows.extend(load_jsonl(_LOG_DIR / f"{d}.jsonl"))
    return all_rows


# ── Metrics computation ──────────────────────────────────────────────


def compute_calibration_curve(
    predictions: list[dict], n_bins: int = 10
) -> tuple[list[float], list[float], list[int]]:
    """Bin predictions by meta_probability and compute actual positive
    rate per bin. Returns (mean_predicted, fraction_positive, counts)."""
    edges = [i / n_bins for i in range(n_bins + 1)]
    mean_pred: list[float] = []
    frac_pos: list[float] = []
    counts: list[int] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        bucket = [
            r for r in predictions
            if (lo <= r.get("meta_probability", 0.5) < hi
                if i < n_bins - 1
                else lo <= r.get("meta_probability", 0.5) <= hi)
        ]
        if not bucket:
            mean_pred.append((lo + hi) / 2)
            frac_pos.append(0.0)
            counts.append(0)
            continue
        probs = [r["meta_probability"] for r in bucket]
        # We don't have ground truth in production logs — approximate
        # from label: AI=1, HUMAN=0, UNCERTAIN=0.5 (excluded from
        # calibration as ambiguous)
        labels_binary = []
        for r in bucket:
            lab = r.get("final_label", "UNCERTAIN")
            if lab == "AI":
                labels_binary.append(1.0)
            elif lab == "HUMAN":
                labels_binary.append(0.0)
            # Skip UNCERTAIN for calibration
        if not labels_binary:
            mean_pred.append(sum(probs) / len(probs))
            frac_pos.append(0.5)
            counts.append(len(bucket))
            continue
        mean_pred.append(sum(probs) / len(probs))
        frac_pos.append(sum(labels_binary) / len(labels_binary))
        counts.append(len(bucket))
    return mean_pred, frac_pos, counts


def compute_ece(mean_pred: list[float], frac_pos: list[float], counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    ece = 0.0
    for mp, fp, n in zip(mean_pred, frac_pos, counts):
        if n > 0:
            ece += (n / total) * abs(mp - fp)
    return ece


def compute_brier(predictions: list[dict]) -> float | None:
    """Brier score using label as truth proxy (UNCERTAIN excluded)."""
    pairs = []
    for r in predictions:
        lab = r.get("final_label", "UNCERTAIN")
        if lab == "AI":
            pairs.append((r["meta_probability"], 1.0))
        elif lab == "HUMAN":
            pairs.append((r["meta_probability"], 0.0))
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def trust_state(
    coverage: float, reliability: float | None, psi: float | None
) -> tuple[str, str]:
    """Return (color, label) for the system trust indicator."""
    if reliability is None:
        return "gray", "NO DATA"
    psi_ok = psi is not None and psi < 0.10
    psi_moderate = psi is not None and psi < 0.25
    if coverage >= 0.60 and reliability >= 0.90 and psi_ok:
        return "green", "GREEN — RELIABLE"
    if coverage >= 0.40 and reliability >= 0.70 and (psi_moderate or psi is None):
        return "orange", "YELLOW — CAUTION"
    return "red", "RED — UNRELIABLE"


# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="AuthentiGuard Control Tower",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("AuthentiGuard")
st.sidebar.caption("Confidence Calibration Dashboard")

available_dates = get_available_dates()
if not available_dates:
    st.error("No prediction logs found in logs/predictions/. Run `python dashboard/generate_demo_data.py` first.")
    st.stop()

date_range = st.sidebar.select_slider(
    "Date range",
    options=available_dates,
    value=(available_dates[0], available_dates[-1]),
)
selected_dates = [
    d for d in available_dates
    if date_range[0] <= d <= date_range[1]
]

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Dates loaded:** {len(selected_dates)}\n\n"
    f"**Log dir:** `{_LOG_DIR.relative_to(_REPO)}`"
)

# ── Load data ────────────────────────────────────────────────────────

all_predictions = load_all_predictions(selected_dates)
daily_metrics = load_json(_METRICS_DIR / "daily_metrics.json")
drift_data = load_json(_METRICS_DIR / "drift.json")
cal_metrics = load_json(_ACCURACY_DIR / "calibration_metrics.json")

n_total = len(all_predictions)
if n_total == 0:
    st.warning("No predictions in the selected date range.")
    st.stop()

# Aggregate stats
label_counts = Counter(r.get("final_label", "UNCERTAIN") for r in all_predictions)
n_definitive = label_counts.get("AI", 0) + label_counts.get("HUMAN", 0)
n_uncertain = label_counts.get("UNCERTAIN", 0)
coverage = n_definitive / n_total if n_total > 0 else 0.0

# Reliability (accuracy on definitive, using label as proxy)
# In production without ground truth, we report coverage + ECE as the
# reliability proxy. For eval-backed reliability, link to ACCURACY.md.
reliability_evals = {}
for name, fname in [("v1", "reliability_eval.json"), ("v2", "reliability_eval_v2.json")]:
    data = load_json(_ACCURACY_DIR / fname)
    if data:
        reliability_evals[name] = data

eval_reliability = None
if reliability_evals:
    latest = list(reliability_evals.values())[-1]
    m = latest.get("metrics_on_definitive_only", {})
    eval_reliability = m.get("reliability_accuracy")

# Latest PSI
latest_psi = None
latest_psi_status = "UNKNOWN"
if drift_data.get("days"):
    latest_day = max(drift_data["days"].keys())
    latest_drift = drift_data["days"][latest_day]
    latest_psi = latest_drift.get("psi")
    latest_psi_status = latest_drift.get("classification", "UNKNOWN")

# ── Header: System Trust State ───────────────────────────────────────

trust_color, trust_label = trust_state(coverage, eval_reliability, latest_psi)

color_map = {"green": "#27ae60", "orange": "#f39c12", "red": "#e74c3c", "gray": "#95a5a6"}
bg = color_map.get(trust_color, "#95a5a6")

st.markdown(
    f"""
    <div style="background:{bg}; padding:1rem 2rem; border-radius:12px;
                text-align:center; margin-bottom:1.5rem;">
        <h1 style="color:white; margin:0; font-size:2rem;">
            AuthentiGuard Control Tower
        </h1>
        <h2 style="color:white; margin:0.3rem 0 0 0; font-size:1.4rem;">
            System Trust: {trust_label}
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI row ──────────────────────────────────────────────────────────

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Predictions", f"{n_total:,}")
kpi2.metric("Coverage", f"{coverage:.1%}", help="Fraction classified as AI or HUMAN (not UNCERTAIN)")
kpi3.metric(
    "Eval Reliability",
    f"{eval_reliability:.1%}" if eval_reliability else "N/A",
    help="Accuracy on definitive predictions (from offline eval)",
)
kpi4.metric(
    "Drift (PSI)",
    f"{latest_psi:.4f}" if latest_psi is not None else "N/A",
    delta=latest_psi_status,
    delta_color="normal" if latest_psi_status == "STABLE" else "inverse",
)

# ── Row 1: Live stream + Label distribution ─────────────────────────

st.markdown("---")
col_stream, col_pie = st.columns([3, 2])

with col_stream:
    st.subheader("Latest Predictions")
    recent = all_predictions[-100:][::-1]
    display_rows = []
    for r in recent:
        ts = r.get("timestamp", "")[:19]
        lab = r.get("final_label", "?")
        prob = r.get("meta_probability", 0)
        margin = r.get("confidence_margin", prob - 0.41)
        zone = r.get("zone", "?")
        ver = r.get("model_version", "?")
        emoji = {"AI": "🤖", "HUMAN": "👤", "UNCERTAIN": "❓"}.get(lab, "?")
        display_rows.append({
            "Time": ts,
            "Label": f"{emoji} {lab}",
            "Score": f"{prob:.3f}",
            "Margin": f"{margin:+.3f}",
            "Zone": zone,
            "Words": r.get("input_length", 0) // 5,
            "Version": ver[-15:],
        })
    st.dataframe(display_rows, use_container_width=True, height=350)

with col_pie:
    st.subheader("Label Distribution")
    fig_pie = px.pie(
        names=list(label_counts.keys()),
        values=list(label_counts.values()),
        color=list(label_counts.keys()),
        color_discrete_map={"AI": "#e74c3c", "HUMAN": "#27ae60", "UNCERTAIN": "#f39c12"},
        hole=0.4,
    )
    fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Row 2: Confidence histogram + Calibration curve ─────────────────

st.markdown("---")
col_hist, col_cal = st.columns(2)

with col_hist:
    st.subheader("Confidence Histogram")
    probs = [r.get("meta_probability", 0.5) for r in all_predictions]
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=probs, nbinsx=20, marker_color="#3498db", name="Score",
    ))
    fig_hist.add_vline(x=0.70, line_dash="dash", line_color="red",
                       annotation_text="AI zone (>=0.70)")
    fig_hist.add_vline(x=0.30, line_dash="dash", line_color="green",
                       annotation_text="HUMAN zone (<=0.30)")
    fig_hist.update_layout(
        xaxis_title="Calibrated Probability (AI)",
        yaxis_title="Count",
        height=350,
        margin=dict(t=30, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_cal:
    st.subheader("Calibration Curve")
    mean_pred, frac_pos, counts = compute_calibration_curve(all_predictions, n_bins=10)
    ece = compute_ece(mean_pred, frac_pos, counts)
    brier = compute_brier(all_predictions)

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=mean_pred, y=frac_pos, mode="markers+lines",
        marker=dict(size=[max(5, min(20, c / 5)) for c in counts]),
        name="Observed",
        text=[f"n={c}" for c in counts],
    ))
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect calibration",
    ))
    subtitle_parts = [f"ECE = {ece:.4f}"]
    if brier is not None:
        subtitle_parts.append(f"Brier = {brier:.4f}")
    fig_cal.update_layout(
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction Actually AI",
        title=dict(text=" | ".join(subtitle_parts), font=dict(size=13)),
        height=350,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # Stored calibration metrics from Stage 2
    if cal_metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("ECE (Stage 2 val)", f"{cal_metrics.get('ece_calibrated', 'N/A')}")
        c2.metric("Brier (calibrated)", f"{cal_metrics.get('brier_calibrated', 'N/A'):.4f}"
                   if isinstance(cal_metrics.get('brier_calibrated'), (int, float)) else "N/A")
        c3.metric("Brier (baseline)", f"{cal_metrics.get('brier_baseline_fixed_weight', 'N/A'):.4f}"
                   if isinstance(cal_metrics.get('brier_baseline_fixed_weight'), (int, float)) else "N/A")

# ── Row 3: Drift trend + Coverage/Reliability per day ────────────────

st.markdown("---")
col_drift, col_cov = st.columns(2)

with col_drift:
    st.subheader("Drift Indicator (PSI)")
    days_data = drift_data.get("days", {})
    if days_data:
        drift_dates = sorted(days_data.keys())
        drift_values = [days_data[d].get("psi", 0) for d in drift_dates]
        drift_status = [days_data[d].get("classification", "?") for d in drift_dates]

        fig_drift = go.Figure()
        colors = ["#27ae60" if s == "STABLE" else "#f39c12" if s == "MODERATE" else "#e74c3c"
                  for s in drift_status]
        fig_drift.add_trace(go.Bar(
            x=drift_dates, y=drift_values,
            marker_color=colors,
            text=drift_status,
            textposition="outside",
        ))
        fig_drift.add_hline(y=0.10, line_dash="dash", line_color="orange",
                            annotation_text="Moderate threshold")
        fig_drift.add_hline(y=0.25, line_dash="dash", line_color="red",
                            annotation_text="Significant threshold")
        fig_drift.update_layout(
            yaxis_title="PSI",
            height=350,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_drift, use_container_width=True)
    else:
        st.info("No drift data available.")

with col_cov:
    st.subheader("Coverage Over Time")
    dm_days = daily_metrics.get("days", {})
    if dm_days:
        dm_dates = sorted(dm_days.keys())
        coverages = []
        uncertain_fracs = []
        for d in dm_dates:
            cb = dm_days[d].get("class_balance", {})
            uf = cb.get("uncertain_fraction", 0)
            coverages.append(1.0 - uf)
            uncertain_fracs.append(uf)

        fig_cov = go.Figure()
        fig_cov.add_trace(go.Scatter(
            x=dm_dates, y=coverages, mode="lines+markers",
            name="Coverage", line=dict(color="#3498db"),
        ))
        fig_cov.add_trace(go.Scatter(
            x=dm_dates, y=uncertain_fracs, mode="lines+markers",
            name="Abstain Rate", line=dict(color="#f39c12", dash="dot"),
        ))
        fig_cov.add_hline(y=0.60, line_dash="dash", line_color="green",
                          annotation_text="GREEN threshold (60%)")
        fig_cov.update_layout(
            yaxis_title="Fraction",
            yaxis_range=[0, 1],
            height=350,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_cov, use_container_width=True)
    else:
        st.info("No daily metrics available.")

# ── Row 4: Confidence margin distribution ────────────────────────────

st.markdown("---")
st.subheader("Confidence Margin Distribution")
st.caption("Distance from 0.41 threshold — positive = AI-leaning, negative = HUMAN-leaning")

margins = [r.get("confidence_margin", r.get("meta_probability", 0.5) - 0.41) for r in all_predictions]
fig_margin = go.Figure()
fig_margin.add_trace(go.Histogram(x=margins, nbinsx=40, marker_color="#8e44ad"))
fig_margin.add_vline(x=0.0, line_dash="dash", line_color="gray",
                     annotation_text="Threshold (0.41)")
fig_margin.add_vline(x=0.29, line_dash="dot", line_color="red",
                     annotation_text="AI zone (0.70)")
fig_margin.add_vline(x=-0.11, line_dash="dot", line_color="green",
                     annotation_text="HUMAN zone (0.30)")
fig_margin.update_layout(
    xaxis_title="Confidence Margin (score - 0.41)",
    yaxis_title="Count",
    height=300,
    margin=dict(t=30, b=40),
)
st.plotly_chart(fig_margin, use_container_width=True)

# ── Row 5: Model version comparison ─────────────────────────────────

st.markdown("---")
st.subheader("Model Version Comparison")

eval_files = {
    "Stage 1 (pre-fit)": "ensemble_test_eval.json",
    "Stage 1 (post-fit)": "ensemble_test_eval.post_fit.json",
    "Stage 2 (meta v1)": "ensemble_test_eval.meta.json",
    "Stage 2 (meta v2)": "ensemble_test_eval_v2.meta.json",
    "Reliability (v1)": "reliability_eval.json",
    "Reliability (v2)": "reliability_eval_v2.json",
}

comparison_rows = []
for label, fname in eval_files.items():
    data = load_json(_ACCURACY_DIR / fname)
    if not data:
        continue
    metrics = data.get("metrics", data.get("metrics_on_definitive_only", {}))
    n = data.get("n_samples", data.get("n_total", "?"))
    coverage_val = data.get("coverage", 1.0)
    comparison_rows.append({
        "Evaluation": label,
        "F1": f"{metrics.get('f1', 'N/A'):.4f}" if isinstance(metrics.get("f1"), (int, float)) else "N/A",
        "Precision": f"{metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get("precision"), (int, float)) else "N/A",
        "Recall": f"{metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get("recall"), (int, float)) else "N/A",
        "AUROC": f"{metrics.get('auroc', 'N/A'):.4f}" if isinstance(metrics.get("auroc"), (int, float)) else "N/A",
        "Reliability": f"{metrics.get('reliability_accuracy', '-')}" if isinstance(metrics.get("reliability_accuracy"), (int, float)) else "-",
        "Coverage": f"{coverage_val:.1%}" if isinstance(coverage_val, (int, float)) else "-",
        "N": str(n),
    })

if comparison_rows:
    st.dataframe(comparison_rows, use_container_width=True)
else:
    st.info("No evaluation artifacts found in ai/text_detector/accuracy/.")

# ── Footer ───────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"Data range: {selected_dates[0]} to {selected_dates[-1]} | "
    f"Predictions: {n_total:,} | "
    f"Auto-refresh: 30s cache | "
    f"Dashboard v1.0"
)
