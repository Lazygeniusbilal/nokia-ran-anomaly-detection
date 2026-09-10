# Nokia RAN Anomaly Detection

Unsupervised anomaly detection for Nokia 5G RAN KPI time-series, using a Conv1D autoencoder and an interactive Streamlit dashboard for drill-down investigation.

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-1.63-ff4b4b)

---

## Overview

**The problem:** Radio Access Networks generate KPI streams — accessibility, drop rate, success rate — for every cell, every 15 minutes. At network scale, that's thousands of cells producing millions of data points a day. Manual, per-metric threshold rules ("alert if drop rate > X%") don't scale: they require a hand-tuned threshold per KPI per cell, miss anomalies that are only unusual in *combination*, and generate constant false positives as normal traffic patterns shift.

**The approach:** train a model to learn what *normal* KPI behavior looks like for a cell, with no labeled anomalies required. A Conv1D autoencoder is trained to compress and reconstruct short windows of real, "normal" KPI data. At inference time, a window that reconstructs poorly — because it doesn't match anything the model learned — is flagged as anomalous. The reconstruction error itself becomes the anomaly score, and a data-driven threshold (rather than a hand-picked one) decides what counts as anomalous.

---

## Demo

![Dashboard screenshot placeholder](docs/demo.gif)
*(screenshot / GIF of the Streamlit dashboard — coming soon)*

**Live demo:** [ADD LINK]

The dashboard lets a viewer pick a cell and KPI, see flagged anomaly windows shaded on the time-series, and drill into any one flagged window to see which KPI(s) drove the anomaly score.

---

## Dataset

- **Source:** Nokia 5G Standalone (SA) RAN KPI data, network-wide scale of **~85M rows across 3,904 cells**.
- **Interval:** 15-minute reporting intervals per cell.
- **KPIs tracked (5):**
  | Column | Description |
  |---|---|
  | `5g_sa_data_session_success_rate` | Data session success rate |
  | `5g_sa_drb_accessibility` | Data Radio Bearer accessibility |
  | `5g_sa_drop_rate` | Call/session drop rate |
  | `5g_sa_ng_accessibility` | NG interface accessibility |
  | `5g_sa_rrc_accessibility` | RRC connection accessibility |
- **Prototype scope:** this repo's current pipeline validates the approach on a locked subset of **100 cells** rather than the full network — real-world RAN data is gap-heavy (cells that don't report for stretches of time, or at all, for a given KPI set), so the pipeline is built mask-aware from the ground up rather than assuming complete data.

---

## Approach

1. **Regularize** each cell's readings onto a fixed 15-minute time grid; a **mask** channel marks which values are real vs. filled-in for missing timestamps.
2. **Scale** each KPI with a `RobustScaler` fit only on real (masked-in) training values, then applied to train and validation alike — no leakage from validation into fitting.
3. **Window** each cell's series into fixed-length sequences: **64 timesteps (~16 hours) per window, stride 16**. Windows with zero real data anywhere in them are dropped — they carry no signal to learn from or score.
4. **Split** chronologically, 80% train / 20% validation — no shuffling, so validation is always later in time than what the model trained on.
5. **Model:** a Conv1D autoencoder. Values and the mask are concatenated as separate input channels, so the network can distinguish "this KPI reads 0.2" from "this KPI is missing" rather than treating them the same.
6. **Loss:** masked MSE — reconstruction error is only counted at real (masked-in) points, so filled-in placeholder values never influence training.
7. **Anomaly score:** per-window reconstruction error, again computed only over real points.
8. **Threshold:** the 95th percentile of *training* scores — training data stands in for "normal," so a window scoring above what 95% of normal windows scored is flagged.

---

## Results

| Metric | Value |
|---|---|
| Final training loss | 0.3468 |
| Final validation loss | 0.3927 |
| Anomaly threshold (95th percentile of train scores) | 1.3881 |
| Validation windows flagged | ~4.7% |

---

## Project Structure

```
src/time_series/
├── data_processing/
│   ├── cleaning.py       # cell selection, regularizing to a fixed time grid + missing-value mask
│   ├── scaling.py        # RobustScaler: fit on train only, apply to train/validation
│   ├── splitting.py      # chronological train/validation split
│   └── windowing.py      # fixed-length sliding windows per cell
├── models/
│   └── conv_autoencoder.py   # Conv1D autoencoder + masked-MSE loss
├── training/
│   └── train.py          # training loop, loss curves
└── evaluation/
    ├── scoring.py         # reconstruction-error scoring, threshold, anomaly flagging
    ├── persistance.py     # save / load trained model weights + fitted scalers
    └── prediction.py      # inference entrypoint — score new/held-out data with a saved model

main.py     # end-to-end training pipeline: data → train → score → save artifacts
app.py      # Streamlit dashboard (overview, time series, anomalies, drill-down)
```

---

## Setup & Usage

### Install dependencies

```bash
# using uv (recommended — this repo ships a uv.lock)
uv sync

# or using pip
pip install -e .
```

> Nokia RAN KPI parquet files are not included in this repo. Place them under `data/unzip/` before running the pipeline.

### Train the model

```bash
uv run main.py
# or: python main.py
```

Runs the full pipeline — data loading, regularization, scaling, windowing, training, scoring — and saves the trained model + scalers to `artifacts/`.

### Run inference

```bash
uv run python -m src.time_series.evaluation.prediction
# or: python src/time_series/evaluation/prediction.py
```

Loads the saved model and scalers from `artifacts/` and scores held-out data without retraining.

### Launch the dashboard

```bash
uv run streamlit run app.py
# or: streamlit run app.py
```

---

## Status

**Built**
- Full data pipeline: cleaning, regularization, masking, scaling, windowing
- Trained Conv1D autoencoder with masked-MSE loss
- Model + scaler persistence (save/load artifacts)
- Standalone inference script for scoring new/held-out data
- Streamlit dashboard: overview metrics, time-series view, anomaly table, and per-window drill-down (zoomed chart + per-KPI error breakdown)
- Cloud deployment on AWS (S3 for data/artifacts, SageMaker for training/serving, CloudWatch for monitoring)
- Experiment tracking with MLflow
- Hyperparameter tuning with Optuna

---

## Tech Stack

**Built with:** Python · PyTorch · Pandas · NumPy · scikit-learn · Streamlit · Plotly

**Planned:** AWS (S3 / SageMaker / CloudWatch) · MLflow · Optuna
