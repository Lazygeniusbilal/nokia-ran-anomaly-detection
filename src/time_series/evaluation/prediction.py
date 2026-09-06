# import dependencies
import os
import torch

from src.time_series.data_processing.cleaning import select_data, regularize_all, kpi_cols
from src.time_series.data_processing.scaling import apply_scalers
from src.time_series.data_processing.splitting import data_split
from src.time_series.data_processing.windowing import make_windows

from src.time_series.evaluation.persistance import load_artifacts
from src.time_series.evaluation.scoring import compute_scores, flag_anomalies


def main():
    # get the model and scalers (trained earlier, loaded here — no re-training)
    model, scalers = load_artifacts(save_dir='artifacts')

    # rebuild the same val split used during training (unseen by the model)
    df = select_data(data_path='data/unzip')
    resultant_df = regularize_all(df=df, kpi_cols=kpi_cols)
    _, val_df = data_split(df=resultant_df, split_ratio=0.8)

    # apply saved scalers only — never fit again on new/val data
    val_df_scaled = apply_scalers(df=val_df, scalers=scalers, kpi_cols=kpi_cols)

    x_val, m_val = make_windows(data=val_df_scaled, kpi_cols=kpi_cols)

    val_scores = compute_scores(
        model, torch.tensor(x_val, dtype=torch.float32), torch.tensor(m_val, dtype=torch.float32)
    )

    # threshold from training run — TODO: save this in save_artifacts instead of hardcoding
    threshold = 1.3881
    val_flags = flag_anomalies(val_scores, threshold)

    print(f"threshold: {threshold:.4f}")
    print(f"windows flagged: {val_flags.sum()} / {len(val_flags)}")


if __name__ == "__main__":
    main()