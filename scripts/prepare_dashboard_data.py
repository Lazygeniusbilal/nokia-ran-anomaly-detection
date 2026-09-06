"""One-time data prep for the Streamlit dashboard.

Regularizes the raw Nokia RAN KPI parquet files (data/unzip/, ~9GB, not committed
to git) into a single small parquet file the deployed app can load directly —
so Streamlit Cloud never needs the raw dataset.

Run this locally whenever the raw data changes, then commit the output:
    python scripts/prepare_dashboard_data.py
"""

import os
import sys
from pathlib import Path

# make `src` importable regardless of the current working directory this is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.time_series.data_processing.cleaning import select_data, regularize_all, kpi_cols

OUTPUT_PATH = "artifacts/regularized_data.parquet"


def main():
    df = select_data(data_path="data/unzip")
    resultant_df = regularize_all(df=df, kpi_cols=kpi_cols)

    # name the index explicitly so the parquet round-trip is unambiguous —
    # regularize_all() leaves it as an unnamed DatetimeIndex
    if resultant_df.index.name is None:
        resultant_df.index.name = "start_time_utc"

    os.makedirs("artifacts", exist_ok=True)
    resultant_df.to_parquet(OUTPUT_PATH)

    print(
        f"Saved {len(resultant_df):,} rows across {resultant_df['object_id'].nunique()} "
        f"cells to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
