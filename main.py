# import dependencies
import torch

# data processing
from src.time_series.data_processing.cleaning import select_data, regularize_all, kpi_cols
from src.time_series.data_processing.scaling import fit_scalers, apply_scalers
from src.time_series.data_processing.splitting import data_split
from src.time_series.data_processing.windowing import make_windows
# model training
from src.time_series.training.train import train_model
# evaluation
from src.time_series.evaluation.scoring import compute_scores, get_threshold, flag_anomalies
from src.time_series.evaluation.persistance import save_artifacts


def main():

    # select the data (100 cells, as select_data locks in)
    df = select_data(data_path='data/unzip')

    # regularize EVERY selected cell (not just one) — full 15-min grid + mask, per cell
    resultant_df = regularize_all(df=df, kpi_cols=kpi_cols)

    # split into train/val (chronological, per cell)
    train_df, val_df = data_split(df=resultant_df, split_ratio=0.8)

    # fit scalers on train only, apply to both
    train_df_scaled, scalers = fit_scalers(train_df=train_df, kpi_cols=kpi_cols)
    val_df_scaled = apply_scalers(df=val_df, scalers=scalers, kpi_cols=kpi_cols)

    # window each split separately
    x_train, m_train = make_windows(data=train_df_scaled, kpi_cols=kpi_cols)
    x_val, m_val = make_windows(data=val_df_scaled, kpi_cols=kpi_cols)

    # train the model
    model, training_loss, validation_loss = train_model(
        x_train=x_train, m_train=m_train, x_val=x_val, m_val=m_val
    )

    # print just the FINAL epoch's numbers, not the whole 150-entry list
    print(f"Final training loss: {training_loss[-1]:.4f}")
    print(f"Final validation loss: {validation_loss[-1]:.4f}")

    # compute the scores
    train_scores = compute_scores(
        model, torch.tensor(x_train, dtype=torch.float32), torch.tensor(m_train, dtype=torch.float32)
    )
    val_scores = compute_scores(
        model, torch.tensor(x_val, dtype=torch.float32), torch.tensor(m_val, dtype=torch.float32)
    )
    threshold = get_threshold(train_scores)
    val_flags = flag_anomalies(val_scores, threshold)

    print(f"threshold: {threshold:.4f}")
    print(f"val windows flagged: {val_flags.sum()} / {len(val_flags)}")
    
    # save the model
    save_artifacts(model= model, scalers= scalers, save_dir='artifacts')


if __name__ == "__main__":
    main()