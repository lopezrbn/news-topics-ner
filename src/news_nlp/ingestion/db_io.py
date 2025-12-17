from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
import zipfile

from news_nlp.config import paths
from news_nlp.db.connection import get_engine
from news_nlp.preprocessing.text_cleaning import clean_text


def _extract_data_from_zip(
    zip_path: Path,
    extract_to: Path,
) -> None:
    """
    Extracts the contents of a ZIP file to a specified directory.

    Parameters
    ----------
    zip_path : Path
        Path to the ZIP file.
    extract_to : Path
        Directory where the contents will be extracted.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    
    # Create raw data directory if it doesn't exist
    paths.DIR_DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Unzip the compressed data file into the raw data directory
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Extraction completed.")


def load_data_into_news_table(
    path: Path,
    source: Literal["train", "test", "prod"],
    data_sep: str = "\t",
) -> None:
    """
    Load one TSV split (train or test) into the `news` table.

    This function:
      1) reads the raw TSV file,
      2) builds the `text` field from title/content,
      3) cleans the text using `clean_text`,
      4) inserts rows into the `news` table.

    Parameters
    ----------
    path_tsv : Path
        Path to the raw TSV file.
    source : {"train", "test", "prod"}
        Value to store in the `source` column of `news`.
    """

    # Load environment variables
    load_dotenv(paths.ENV_FILE)

    if not path.exists():
        _extract_data_from_zip(paths.DATA_COMPRESSED, paths.DIR_DATA_RAW)
        if not path.exists():
            raise FileNotFoundError(f"TSV file not found: {path}")

    print(f"Reading raw TSV from {path} (source='{source}')")

    # Read the raw TSV/CSV file
    df_raw = pd.read_csv(path, sep=data_sep)

    # Basic sanity checks on expected columns
    required_cols = ["title", "content"]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    # Build text column
    df_raw["title"] = df_raw["title"].fillna("")
    df_raw["content"] = df_raw["content"].fillna("")
    df_raw["text_raw"] = df_raw["title"] + ". " + df_raw["content"]

    # Clean text using the shared cleaning function
    df_raw["text"] = df_raw["text_raw"].apply(clean_text)

    # Build the dataframe aligned with the "news" table schema
    # "id_news" column is SERIAL in DB, so we do not set it here
    # "ingested_at" column is set by the DB default (NOW()), so not need either
    df_news = df_raw[["title", "content", "text"]].copy()
    df_news.insert(0, "source", source)

    engine = get_engine()

    df_news.to_sql(
        "news",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1_000,
        method="multi",
    )

    print(f"Inserted {len(df_news)} rows from {path} with source='{source}'.")