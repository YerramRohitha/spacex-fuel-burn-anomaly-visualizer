import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils import (
    prepare_dataframe,
    compute_expected_curve,
    detect_anomalies,
    summarize_anomalies,
)

st.set_page_config(page_title="SpaceX Fuel Burn Anomaly Visualizer", layout="wide")
st.title("🚀 SpaceX Fuel Burn Anomaly Visualizer")

st.markdown(
    '''
This app visualizes rocket **fuel burn** and flags **anomalies** by comparing actual burn against an expected (nominal) curve.
- Works with the included sample telemetry or your own CSV.
- Minimum columns: `time_s` and **one** of: `burn_rate_kg_s` or `fuel_mass_kg` (we'll derive burn rate from mass if needed).
- Optional: `stage` (1/2), `event` (e.g., MECO, SECO).
'''
)

with st.sidebar:
    st.header("Settings")
    smoothing_window = st.slider("Smoothing window (seconds)", 1, 30, 8, 1)
    z_thresh = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
    iso_frac = st.slider("IsolationForest contamination (fraction)", 0.0, 0.2, 0.02, 0.01)
    use_isolation_forest = st.checkbox("Use IsolationForest (in addition to z-score)", True)
    show_expected = st.checkbox("Show expected (nominal) curve", True)
    show_residuals = st.checkbox("Show residuals panel", False)
    st.divider()
    st.caption("Tip: Tweak thresholds to see how anomaly flags change.")

st.subheader("1) Load Telemetry")
uploaded = st.file_uploader("Upload CSV (or use the bundled sample)", type=["csv"])
if uploaded is None:
    st.info("Using bundled sample data: `data/sample_spacex_telemetry.csv`")
    df = pd.read_csv("data/sample_spacex_telemetry.csv")
else:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

# Prepare & compute expected curve
df = prepare_dataframe(df)
df_expected = compute_expected_curve(df, smoothing_window=int(smoothing_window))

# Detect anomalies
df_flagged = detect_anomalies(
    df_expected,
    z_thresh=z_thresh,
    use_iforest=use_isolation_forest,
    contamination=max(iso_frac, 1e-6),
)

# Summary
summary = summarize_anomalies(df_flagged)

# ------------------------------
# Plots
# ------------------------------
st.subheader("2) Fuel Burn Over Time")

fig = go.Figure()

# Actual burn
fig.add_trace(
    go.Scatter(
        x=df_flagged["time_s"],
        y=df_flagged["burn_rate_kg_s"],
        mode="lines",
        name="Actual burn rate",
    )
)

# Expected burn
if show_expected:
    fig.add_trace(
        go.Scatter(
            x=df_flagged["time_s"],
            y=df_flagged["expected_burn_rate"],
            mode="lines",
            name="Expected (nominal)",
        )
    )

# Anomaly points
anom = df_flagged[df_flagged["is_anomaly"] == True]
fig.add_trace(
    go.Scatter(
        x=anom["time_s"],
        y=anom["burn_rate_kg_s"],
        mode="markers",
        name="Anomalies",
        marker=dict(size=8, symbol="x"),
    )
)

# Stage shading
if "stage" in df_flagged.columns and df_flagged["stage"].nunique() > 1:
    # Add vertical regions per stage
    for stage_val in sorted(df_flagged["stage"].dropna().unique()):
        stage_df = df_flagged[df_flagged["stage"] == stage_val]
        if stage_df.empty:
            continue
        x0, x1 = stage_df["time_s"].min(), stage_df["time_s"].max()
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=None,
            line_width=1,
            annotation_text=f"Stage {int(stage_val)}",
            annotation_position="top left",
            layer="below"
        )

# Event markers
if "event" in df_flagged.columns:
    max_y = float(np.nanmax(df_flagged["burn_rate_kg_s"])) if len(df_flagged) else 0.0
    for _, r in df_flagged.dropna(subset=["event"]).iterrows():
        fig.add_vline(x=float(r["time_s"]), line_dash="dot")
        fig.add_annotation(x=float(r["time_s"]), y=max_y, text=str(r["event"]), showarrow=False, yshift=10)

fig.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Burn rate (kg/s)",
    legend_title_text="Series",
    height=520,
)

st.plotly_chart(fig, use_container_width=True)

# Residuals
if show_residuals:
    st.subheader("Residuals (Actual - Expected)")
    res_fig = go.Figure()
    res_fig.add_trace(
        go.Scatter(
            x=df_flagged["time_s"],
            y=df_flagged["residual"],
            mode="lines",
            name="Residual",
        )
    )
    res_fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(res_fig, use_container_width=True)

# ------------------------------
# Tables
# ------------------------------
st.subheader("3) Anomalies Table")
st.dataframe(summary, use_container_width=True)

st.download_button(
    label="⬇️ Download anomalies as CSV",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="anomalies_summary.csv",
    mime="text/csv",
)

st.subheader("4) Data Preview")
st.dataframe(df_flagged.head(100), use_container_width=True)

st.caption("Built with ❤️ using Streamlit + Plotly + scikit-learn")