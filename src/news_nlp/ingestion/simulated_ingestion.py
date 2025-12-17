from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def _parse_iso_dt(value: str) -> datetime:
    # Airflow logical_date is tz-aware; we keep tz awareness end-to-end.
    return datetime.fromisoformat(value)


def load_fraction_prod_into_news_table(
    tsv_path: Path,
    start_logical_dt_iso: str,
    period_seconds: int,
    fraction_per_run: float,
    logical_date,  # pendulum.DateTime from Airflow context
    *,
    data_sep: str = "\t",
) -> int:
    """
    Load a deterministic slice of the TSV into the news table using the existing loader,
    storing source='prod'.

    Returns the number of rows selected (attempted).
    """
    if not (0.0 < fraction_per_run <= 1.0):
        raise ValueError("fraction_per_run must be in (0, 1].")

    if period_seconds <= 0:
        raise ValueError("period_seconds must be > 0.")

    start_dt = _parse_iso_dt(start_logical_dt_iso)

    # logical_date is typically a pendulum.DateTime (tz-aware). Convert to python datetime (keeps tz).
    run_dt = logical_date if isinstance(logical_date, datetime) else logical_date.naive().replace(tzinfo=logical_date.tzinfo)

    delta_seconds = (run_dt - start_dt).total_seconds()
    run_index = math.floor(delta_seconds / period_seconds)

    if run_index < 0:
        print("Run is before START_LOGICAL_DT; selecting 0 rows.")
        return 0

    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    # Read full TSV once
    df_raw = pd.read_csv(tsv_path, sep=data_sep)

    total_rows = len(df_raw)
    if total_rows == 0:
        print("TSV has 0 rows; selecting 0 rows.")
        return 0

    batch_size = max(1, math.ceil(total_rows * fraction_per_run))

    start_row = run_index * batch_size
    end_row = min(start_row + batch_size, total_rows)

    if start_row >= total_rows:
        print("Dataset exhausted; selecting 0 rows.")
        return 0

    df_slice = df_raw.iloc[start_row:end_row].copy()

    # Write slice to a temporary TSV file for ingestion
    path_df_slice = "temp_slice.tsv"
    df_slice.to_csv(path_df_slice, sep=data_sep, index=False)

    print(
        f"Selected slice [{start_row}, {end_row}) out of {total_rows} "
        f"(batch_size={batch_size}, run_index={run_index}, logical_date={logical_date})."
    )

    from news_nlp.ingestion.db_io import load_data_into_news_table

    load_data_into_news_table(path=path_df_slice, source="prod", data_sep=data_sep)

    # Clean up temporary file
    os.remove(path_df_slice)

    return len(df_slice)
