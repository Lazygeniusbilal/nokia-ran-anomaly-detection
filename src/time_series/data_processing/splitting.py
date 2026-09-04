import pandas as pd

def data_split(df: pd.DataFrame, split_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ONE cell's time-ordered data chronologically.
    Assumes df is already sorted by time (single cell, no shuffling).
    First `split_ratio` fraction -> train, remainder -> val.
    """
    n = int(len(df) * split_ratio)
    train_df = df.iloc[:n]
    val_df = df.iloc[n:]
    return train_df, val_df