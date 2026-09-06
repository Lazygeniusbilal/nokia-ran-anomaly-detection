import numpy as np
import pandas as pd


def make_windows(data: pd.DataFrame, kpi_cols: list[str], window: int = 64, stride: int = 16):
    scaled_cols = [c + '_scaled' for c in kpi_cols]
    mask_cols   = [c + '_mask' for c in kpi_cols]

    X_windows, M_windows = [], []
    for cell_id, group in data.groupby('object_id'):
        group = group.sort_index()
        values = group[scaled_cols].values
        masks  = group[mask_cols].values
        T = len(group)
        for start in range(0, T - window + 1, stride):
            window_mask = masks[start:start+window]
            if window_mask.sum() == 0:
                continue  # entirely fake/missing window — no real data to reconstruct or score
            X_windows.append(values[start:start+window])
            M_windows.append(window_mask)

    return np.array(X_windows), np.array(M_windows)