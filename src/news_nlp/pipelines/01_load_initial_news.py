from pathlib import Path
import sys
BASE_DIR = str(Path(__file__).resolve().parents[1])
if BASE_DIR not in sys.path:
    print(f"Adding {BASE_DIR} to sys.path")
    sys.path.insert(0, BASE_DIR)
import config.paths as paths
from db.connection import get_engine

from typing import Literal

import pandas as pd
from dotenv import load_dotenv


def load_data_into_news_table(
    path,
    source: Literal["train", "test"],
) -> None:
    """
    Load one news split (train or test) from a parquet file into the `news` table
    using pandas.to_sql.

    Parameters
    ----------
    path :
        Path-like object to the parquet file containing the cleaned news.
    source : Literal["train", "test"]
        Value to store in the `source` column (e.g. 'train' or 'test').
    """

    df = pd.read_parquet(path)

    # Basic sanity checks on expected columns
    required_columns = ["title", "content", "text"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe from {path}")

    # Add the columns missing in the dataframe but expected by the `news` table
    df["source"] = source

    # Keep only the columns expected by the `news` table
    cols_to_keep = ["source", "title", "content", "text"]
    df_result = df[cols_to_keep].copy()

    engine = get_engine()

    df_result.to_sql(
        "news",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1_000,
        method="multi",
    )

    print(f"Inserted {len(df_result)} rows from {path} with source='{source}'")


def main() -> None:
    """
    Entry point to populate the `news` table from processed parquet files.
    """
    # Load environment variables from .env in the project root
    dotenv_path = paths.ENV_FILE
    load_dotenv(dotenv_path)

    # Load train split
    load_data_into_news_table(
        path=paths.DF_TRAIN_CLEAN,
        source="train",
    )

    # Load test split
    load_data_into_news_table(
        path=paths.DF_TEST_CLEAN,
        source="test",
    )


if __name__ == "__main__":
    main()
