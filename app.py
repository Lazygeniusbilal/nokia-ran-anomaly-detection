# app.py
"""Streamlit dashboard for Nokia RAN 5G KPI anomaly detection.

UI/UX layer only — the pipeline (cleaning, scaling, windowing, scoring, persistence)
is imported and called as-is, never modified here.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.time_series.data_processing.cleaning import select_data, regularize_all, kpi_cols
from src.time_series.data_processing.scaling import apply_scalers
from src.time_series.data_processing.windowing import make_windows
from src.time_series.evaluation.persistance import load_artifacts
from src.time_series.evaluation.scoring import compute_scores, flag_anomalies

WINDOW = 64                    # must match training
STRIDE = 16                    # must match training
DEFAULT_THRESHOLD = 1.3881     # from the training run in main.py
DRILLDOWN_PAD_STEPS = WINDOW // 2  # extra timesteps of context shown before/after a drilled-down window

COLORS = {
    "primary": "#124191",              # Nokia-blue accent, used for all "normal" series
    "anomaly": "#E5484D",
    "anomaly_fill": "rgba(229, 72, 77, 0.18)",
    "grid": "#E3E7EE",
}

ACRONYMS = {"5g": "5G", "sa": "SA", "drb": "DRB", "rrc": "RRC", "ng": "NG"}


def pretty_kpi(name: str) -> str:
    """'5g_sa_drb_accessibility' -> '5G SA DRB Accessibility' for display only."""
    return " ".join(ACRONYMS.get(tok, tok.capitalize()) for tok in name.split("_"))


st.set_page_config(
    page_title="RAN Anomaly Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1 {font-size: 2.1rem; font-weight: 700;}
    .stTabs [data-baseweb="tab"] p {font-size: 1.05rem; font-weight: 500;}
    div[data-testid="stMetricValue"] {font-size: 1.75rem;}
    div[data-testid="stMetricLabel"] {font-size: 0.85rem; color: #5B6472;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model_and_scalers():
    return load_artifacts(save_dir="artifacts")


@st.cache_data(show_spinner=False)
def get_regularized_data():
    df = select_data(data_path="data/unzip")
    return regularize_all(df=df, kpi_cols=kpi_cols)


def _kept_window_starts(masks: np.ndarray, T: int, window: int, stride: int) -> list[int]:
    """Mirror make_windows()'s own "skip if fully missing" filter, without importing its
    internals, so each row make_windows() returns can be matched back to its real start
    position. A naive `i * stride` offset silently drifts out of sync as soon as any
    earlier window in that cell gets skipped."""
    starts = []
    for start in range(0, T - window + 1, stride):
        if masks[start : start + window].sum() == 0:
            continue
        starts.append(start)
    return starts


@st.cache_data(show_spinner=False)
def compute_all_scores(_model, _scalers, full_df: pd.DataFrame) -> pd.DataFrame:
    """Score every window for every cell once, so the dashboard can slice/aggregate freely."""
    mask_cols = [c + "_mask" for c in kpi_cols]
    records = []
    for cell_id, group in full_df.groupby("object_id"):
        cell_df = group.sort_index().copy()
        cell_df_scaled = apply_scalers(df=cell_df, scalers=_scalers, kpi_cols=kpi_cols)
        x, m = make_windows(data=cell_df_scaled, kpi_cols=kpi_cols, window=WINDOW, stride=STRIDE)
        if len(x) == 0:
            continue
        scores = compute_scores(
            _model, torch.tensor(x, dtype=torch.float32), torch.tensor(m, dtype=torch.float32)
        )
        timestamps = cell_df.index
        starts = _kept_window_starts(cell_df_scaled[mask_cols].to_numpy(), len(cell_df_scaled), WINDOW, STRIDE)
        for start_idx, score in zip(starts, scores):
            end_idx = start_idx + WINDOW - 1
            records.append(
                {"object_id": cell_id, "start": timestamps[start_idx], "end": timestamps[end_idx], "score": float(score)}
            )
    return pd.DataFrame(records)


def compute_window_kpi_breakdown(
    model, cell_df_scaled: pd.DataFrame, window_start: pd.Timestamp, window: int = WINDOW
) -> pd.DataFrame:
    """Per-KPI reconstruction error for ONE specific window — the drill-down "why".

    Additive, standalone computation: reuses the already-loaded model to reconstruct
    just this one window and breaks the masked squared error down by KPI channel,
    instead of summing across channels the way compute_scores() does. Does not call
    or change compute_scores() — this is a separate view onto the same math.
    """
    scaled_cols = [c + "_scaled" for c in kpi_cols]
    mask_cols = [c + "_mask" for c in kpi_cols]

    start_idx = cell_df_scaled.index.get_loc(window_start)
    x_window = cell_df_scaled[scaled_cols].to_numpy()[start_idx : start_idx + window]
    m_window = cell_df_scaled[mask_cols].to_numpy()[start_idx : start_idx + window]

    x_t = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0)
    m_t = torch.tensor(m_window, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        recon = model(x_t, m_t)
        error = (((recon - x_t) ** 2) * m_t).squeeze(0).numpy()  # (window, n_features)

    rows = []
    for k, kpi in enumerate(kpi_cols):
        mask_sum = float(m_window[:, k].sum())
        kpi_error = float(error[:, k].sum() / mask_sum) if mask_sum > 0 else np.nan
        rows.append({"kpi": kpi, "error": kpi_error, "coverage": mask_sum / window})
    return pd.DataFrame(rows)


with st.spinner("Loading model and scalers..."):
    model, scalers = get_model_and_scalers()

with st.spinner("Loading and regularizing KPI data..."):
    full_df = get_regularized_data()

with st.spinner("Scoring every cell (Conv1D autoencoder forward pass)..."):
    all_scores_df = compute_all_scores(model, scalers, full_df)

if all_scores_df.empty:
    st.error("No scoreable windows were produced. Check the data pipeline / artifacts.")
    st.stop()

available_cells = sorted(all_scores_df["object_id"].unique())

# ---------------------------------------------------------------- sidebar --
st.sidebar.title("📡 Controls")
selected_cell = st.sidebar.selectbox("Cell (object_id)", available_cells)
selected_kpi = st.sidebar.selectbox("KPI to visualize", kpi_cols, format_func=pretty_kpi)

st.sidebar.markdown("---")

# base the slider range on the 99.5th percentile, not the raw max — a handful of
# near-fully-missing windows can produce extreme outlier scores that would make the
# slider step too coarse for the range where thresholds actually matter.
score_p995 = float(np.percentile(all_scores_df["score"], 99.5))
n_outliers_above_range = int((all_scores_df["score"] > score_p995).sum())
slider_max = round(max(score_p995 * 1.2, DEFAULT_THRESHOLD * 1.5), 4)
slider_step = max(round(slider_max / 500, 4), 0.001)
default_threshold = min(max(DEFAULT_THRESHOLD, 0.0), slider_max)

threshold = st.sidebar.slider(
    "Anomaly threshold",
    min_value=0.0,
    max_value=slider_max,
    value=round(default_threshold, 4),
    step=slider_step,
    help="Windows with reconstruction error above this line are flagged as anomalies. "
    "Drag it to see the flagged count react live.",
)
st.sidebar.caption(
    f"Training-run default was {DEFAULT_THRESHOLD}. "
    f"({n_outliers_above_range} extreme outlier window(s) score above this slider's range "
    "and stay flagged regardless.)"
)

st.sidebar.markdown("---")
st.sidebar.caption("Nokia RAN 5G SA KPIs · Conv1D autoencoder · reconstruction-error anomaly detection")

# ------------------------------------------------------------ derived data --
all_scores_df = all_scores_df.assign(
    flagged=flag_anomalies(all_scores_df["score"].to_numpy(), threshold)
)

cell_df = full_df[full_df["object_id"] == selected_cell].sort_index()
cell_scores_df = (
    all_scores_df[all_scores_df["object_id"] == selected_cell]
    .sort_values("start")
    .reset_index(drop=True)
)
# scaled view of this one cell only, needed for the drill-down's per-window reconstruction —
# cheap to recompute for a single cell, reuses apply_scalers exactly as the pipeline does
cell_df_scaled = apply_scalers(df=cell_df.copy(), scalers=scalers, kpi_cols=kpi_cols)

# --------------------------------------------------------- header + metrics --
st.title("Nokia RAN — Unsupervised Anomaly Detection")
st.caption("Conv1D autoencoder trained on 5G SA RAN KPIs · anomalies are windows the model reconstructs poorly")

total_windows = len(all_scores_df)
pct_flagged = 100 * all_scores_df["flagged"].mean()
n_cells = all_scores_df["object_id"].nunique()
avg_error = all_scores_df["score"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total windows scored", f"{total_windows:,}")
m2.metric("% flagged as anomalies", f"{pct_flagged:.1f}%")
m3.metric("Cells covered", f"{n_cells}")
m4.metric("Avg reconstruction error", f"{avg_error:.4f}")

st.write("")

tab_overview, tab_timeseries, tab_anomalies, tab_drilldown = st.tabs(
    ["🏠 Overview", "📈 Time Series", "🚨 Anomalies", "🔎 Drill-down"]
)

# ------------------------------------------------------------- Overview tab --
with tab_overview:
    with st.expander("ℹ️ How this works", expanded=False):
        st.markdown(
            """
            A **Conv1D autoencoder** is trained to compress and reconstruct short windows
            (64 timesteps ≈ 16 hours) of "normal" 5G KPI behavior for each cell.

            - The model only ever sees real (non-missing) KPI readings and learns to reproduce them.
            - For every window, we measure the **reconstruction error** — how different the
              model's rebuilt version is from the real values.
            - A window that looks like typical traffic reconstructs cleanly (**low error**).
              A window with unusual behavior — an outage, a spike, a drop in accessibility —
              reconstructs poorly (**high error**), because the model never learned that pattern.
            - Windows whose error crosses the **threshold** (set in the sidebar) are flagged
              as anomalies. No labeled anomaly examples were used anywhere — the model only
              ever learned what "normal" looks like.
            """
        )

    st.subheader("Reconstruction error distribution — all cells")
    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Histogram(x=all_scores_df["score"], nbinsx=60, marker_color=COLORS["primary"], name="windows")
    )
    hist_fig.add_vline(
        x=threshold, line_dash="dash", line_color=COLORS["anomaly"],
        annotation_text=f"threshold = {threshold:.3f}", annotation_position="top right",
    )
    hist_fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Reconstruction error", yaxis_title="Window count",
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    hist_fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
    hist_fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"])
    st.plotly_chart(hist_fig, use_container_width=True)

    st.subheader("Cells with the most flagged windows")
    per_cell = (
        all_scores_df.groupby("object_id")["flagged"]
        .agg(flagged_windows="sum", total_windows="count")
        .assign(flag_rate=lambda d: 100 * d["flagged_windows"] / d["total_windows"])
        .sort_values("flagged_windows", ascending=False)
        .head(10)
    )
    if per_cell["flagged_windows"].sum() == 0:
        st.info("No windows are flagged at the current threshold.")
    else:
        st.dataframe(
            per_cell.style.format({"flag_rate": "{:.1f}%"}).background_gradient(
                cmap="Reds", subset=["flagged_windows"]
            ),
            use_container_width=True,
        )

# ---------------------------------------------------------- Time Series tab --
with tab_timeseries:
    st.subheader(f"{pretty_kpi(selected_kpi)} — {selected_cell}")

    if cell_scores_df.empty:
        st.warning("Not enough data for this cell to form a window.")
    else:
        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=cell_df.index, y=cell_df[selected_kpi],
                mode="lines", line=dict(color=COLORS["primary"], width=1.4),
                name=pretty_kpi(selected_kpi),
            )
        )
        for _, row in cell_scores_df[cell_scores_df["flagged"]].iterrows():
            fig_ts.add_vrect(x0=row["start"], x1=row["end"], fillcolor=COLORS["anomaly_fill"], line_width=0)
        # legend proxy for the shaded anomaly regions (vrects don't appear in the legend)
        fig_ts.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=COLORS["anomaly"], symbol="square"),
                name="Anomaly window",
            )
        )
        fig_ts.update_layout(
            height=440, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Time", yaxis_title=pretty_kpi(selected_kpi),
            plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_ts.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
        fig_ts.update_yaxes(showgrid=True, gridcolor=COLORS["grid"])
        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption(
            f"{int(cell_scores_df['flagged'].sum())} / {len(cell_scores_df)} windows flagged "
            f"for this cell at threshold {threshold:.3f}"
        )

# ------------------------------------------------------------ Anomalies tab --
with tab_anomalies:
    st.subheader(f"Reconstruction error per window — {selected_cell}")

    if cell_scores_df.empty:
        st.warning("Not enough data for this cell to form a window.")
    else:
        marker_colors = np.where(cell_scores_df["flagged"], COLORS["anomaly"], COLORS["primary"])
        fig_err = go.Figure()
        fig_err.add_trace(
            go.Scatter(
                x=cell_scores_df.index, y=cell_scores_df["score"],
                mode="lines+markers", line=dict(color=COLORS["primary"], width=1.2),
                marker=dict(size=6, color=marker_colors), name="reconstruction error",
            )
        )
        fig_err.add_hline(
            y=threshold, line_dash="dash", line_color=COLORS["anomaly"],
            annotation_text=f"threshold = {threshold:.3f}", annotation_position="top left",
        )
        fig_err.update_layout(
            height=360, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Window index", yaxis_title="Reconstruction error",
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        )
        fig_err.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
        fig_err.update_yaxes(showgrid=True, gridcolor=COLORS["grid"])
        st.plotly_chart(fig_err, use_container_width=True)

        st.subheader("Flagged anomaly windows")
        flagged_df = cell_scores_df.loc[cell_scores_df["flagged"], ["start", "end", "score"]].reset_index(drop=True)
        if flagged_df.empty:
            st.success("No anomalies flagged for this cell at the current threshold.")
        else:
            st.dataframe(
                flagged_df.style.format({"score": "{:.4f}"}).background_gradient(cmap="Reds", subset=["score"]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"{len(flagged_df)} / {len(cell_scores_df)} windows flagged (threshold = {threshold:.3f})")
            st.caption("Head to the **🔎 Drill-down** tab to inspect any one of these windows in detail.")

# ------------------------------------------------------------- Drill-down tab --
with tab_drilldown:
    st.subheader(f"Drill into a flagged window — {selected_cell}")

    flagged_for_drilldown = (
        cell_scores_df[cell_scores_df["flagged"]].sort_values("score", ascending=False).reset_index(drop=True)
    )

    if flagged_for_drilldown.empty:
        st.info(
            "No anomalies are flagged for this cell at the current threshold. "
            "Lower the threshold in the sidebar, or pick another cell, to drill into a specific window."
        )
    else:
        def _format_window_option(i: int) -> str:
            row = flagged_for_drilldown.loc[i]
            return f"{row['start']:%b %d, %H:%M} → {row['end']:%b %d, %H:%M}   ·   score {row['score']:.3f}"

        selected_idx = st.selectbox(
            "Flagged window (sorted by severity, most anomalous first)",
            options=list(range(len(flagged_for_drilldown))),
            format_func=_format_window_option,
            key=f"drilldown_window_{selected_cell}",
        )
        window_row = flagged_for_drilldown.loc[selected_idx]
        window_start, window_end, window_score = window_row["start"], window_row["end"], window_row["score"]

        breakdown_df = compute_window_kpi_breakdown(model, cell_df_scaled, window_start, WINDOW)
        breakdown_df["kpi_pretty"] = breakdown_df["kpi"].map(pretty_kpi)

        valid_breakdown = breakdown_df.dropna(subset=["error"])
        if not valid_breakdown.empty:
            top = valid_breakdown.loc[valid_breakdown["error"].idxmax()]
            st.info(
                f"This window was flagged primarily due to **{pretty_kpi(top['kpi'])}** "
                f"(per-KPI reconstruction error {top['error']:.3f}) — well above what the model "
                f"expects for normal behavior. Overall window score: **{window_score:.3f}** "
                f"(threshold {threshold:.3f})."
            )

        col_zoom, col_breakdown = st.columns([3, 2])

        with col_zoom:
            st.markdown(f"**{pretty_kpi(selected_kpi)} — zoomed to this window**")
            pad = pd.Timedelta(minutes=15 * DRILLDOWN_PAD_STEPS)
            zoomed = cell_df.loc[(cell_df.index >= window_start - pad) & (cell_df.index <= window_end + pad)]

            fig_zoom = go.Figure()
            fig_zoom.add_trace(
                go.Scatter(
                    x=zoomed.index, y=zoomed[selected_kpi], mode="lines+markers",
                    line=dict(color=COLORS["primary"], width=1.6), marker=dict(size=4),
                    name=pretty_kpi(selected_kpi),
                )
            )
            fig_zoom.add_vrect(
                x0=window_start, x1=window_end, fillcolor=COLORS["anomaly_fill"], line_width=0,
                annotation_text="flagged window", annotation_position="top left",
            )
            fig_zoom.update_layout(
                height=380, margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Time", yaxis_title=pretty_kpi(selected_kpi),
                plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
            )
            fig_zoom.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
            fig_zoom.update_yaxes(showgrid=True, gridcolor=COLORS["grid"])
            st.plotly_chart(fig_zoom, use_container_width=True)
            st.caption(f"Shaded region is the flagged window; shown with ~{DRILLDOWN_PAD_STEPS * 15} minutes of context on each side.")

        with col_breakdown:
            st.markdown("**Why was it flagged? Error by KPI**")
            plot_df = breakdown_df.sort_values("error", ascending=True, na_position="first")
            bar_colors = plot_df["error"].fillna(0)
            fig_bar = go.Figure(
                go.Bar(
                    x=bar_colors, y=plot_df["kpi_pretty"], orientation="h",
                    marker=dict(color=bar_colors, colorscale="Reds"),
                    text=[f"{e:.3f}" if pd.notna(e) else "no data" for e in plot_df["error"]],
                    textposition="outside",
                )
            )
            fig_bar.update_layout(
                height=280, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Reconstruction error", plot_bgcolor="white", paper_bgcolor="white",
            )
            fig_bar.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
            st.plotly_chart(fig_bar, use_container_width=True)

            table_df = breakdown_df[["kpi_pretty", "error", "coverage"]].rename(
                columns={"kpi_pretty": "KPI", "error": "Error", "coverage": "Real-data coverage"}
            )
            st.dataframe(
                table_df.style.format({"Error": "{:.4f}", "Real-data coverage": "{:.0%}"}).background_gradient(
                    cmap="Reds", subset=["Error"]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("\"Real-data coverage\" is the share of readings in this window that weren't missing for that KPI.")
