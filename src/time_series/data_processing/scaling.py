# import dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def fit_scalers(train_df, kpi_cols):
    scalers = {}
    for col in kpi_cols:
        mask_col = col + '_mask'
        real_values = train_df.loc[train_df[mask_col] == 1, [col]]
        scaler = RobustScaler()          # new scaler per column
        scaler.fit(real_values)
        scalers[col] = scaler            # keep it
        train_df[col + '_scaled'] = scaler.transform(train_df[[col]])  # transform ALL rows
    return train_df, scalers

def apply_scalers(df: pd.DataFrame, scalers: dict[str, RobustScaler], kpi_cols: list[str]) -> pd.DataFrame:
    """
    Apply already-fitted scalers (from fit_scalers on train_df) to any dataframe —
    used for val_df, or later for new/live data. Never fits here, only transforms.
    """
    for col in kpi_cols:
        scaler = scalers[col]
        df[col + '_scaled'] = scaler.transform(df[[col]])
    return df