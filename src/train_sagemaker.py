import os
import torch

from time_series.data_processing.cleaning import select_data, regularize_all, kpi_cols
from time_series.data_processing.scaling import fit_scalers, apply_scalers
from time_series.data_processing.splitting import data_split
from time_series.data_processing.windowing import make_windows
from time_series.training.train import train_model
from time_series.evaluation.persistance import save_artifacts

def main():
    data_path = os.environ.get("SM_CHANNEL_TRAIN", "data/unzip")
    model_dir = os.environ.get("SM_MODEL_DIR", "artifacts")

    df = select_data(data_path=data_path)
    resultant_df = regularize_all(df=df, kpi_cols=kpi_cols)
    train_df, val_df = data_split(df=resultant_df, split_ratio=0.8)
    train_df_scaled, scalers = fit_scalers(train_df=train_df, kpi_cols=kpi_cols)
    val_df_scaled = apply_scalers(df=val_df, scalers=scalers, kpi_cols=kpi_cols)
    x_train, m_train = make_windows(data=train_df_scaled, kpi_cols=kpi_cols)
    x_val, m_val = make_windows(data=val_df_scaled, kpi_cols=kpi_cols)

    model, training_loss, validation_loss = train_model(
        x_train=x_train, m_train=m_train, x_val=x_val, m_val=m_val
    )
    print(f"Final training loss: {training_loss[-1]:.4f}")
    print(f"Final validation loss: {validation_loss[-1]:.4f}")

    save_artifacts(model=model, scalers=scalers, save_dir=model_dir)

if __name__ == "__main__":
    main()