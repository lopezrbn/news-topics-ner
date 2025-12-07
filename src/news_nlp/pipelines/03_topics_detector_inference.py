from __future__ import annotations

import argparse
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from config import paths
from news_nlp.db.connection import get_engine
from news_nlp.topics_detector.model import load_topic_pipeline
from news_nlp.topics_detector.inference import predict_topics_for_texts
from news_nlp.topics_detector.tables import build_topics_per_news_df
from news_nlp.topics_detector.db_io import (
    get_active_run_id,
    delete_existing_assignments,
    load_news_to_process,
    insert_topics_per_news_df,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the topics inference pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Topics detector inference pipeline (parametrizable by sources, mode, run_id)."
    )

    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help=(
            "Comma-separated list of sources to process "
            "(e.g. 'train', 'train, test', 'prod', 'all'). "
            "Default: 'all'."
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["incremental", "overwrite"],
        default="incremental",
        help=(
            "Inference mode: "
            "'incremental' (only news without topics for this run) or "
            "'overwrite' (delete previous assignments for this run+sources, then recompute). "
            "Default: 'incremental'."
        ),
    )

    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help=(
            "Run id (id_run) in topics_model_training_runs. "
            "If not provided, the active run (is_active=true) will be used."
        ),
    )

    return parser.parse_args()


def parse_sources_arg(sources: str) -> Optional[List[str]]:
    """
    Parse a comma-separated sources string.

    Examples
    --------
    "train"        -> ["train"]
    "train, test"   -> ["train", "test"]
    "all"          -> None  (meaning: no filtering by source)

    Returns
    -------
    list[str] or None
        List of sources to filter by, or None for all sources.
    """
    sources = sources.strip().lower()
    if sources == "all":
        return None

    parts = [s.strip() for s in sources.split(",") if s.strip()]
    if not parts:
        return None

    return parts


def run_topics_detector_inference_job(
    id_run: int,
    sources: Optional[Iterable[str]],
    mode: str,
) -> None:
    """
    Run topics inference for a given run and subset of news, in the requested mode.

    Steps:
      1) If mode='overwrite', delete previous assignments for this run + sources.
      2) Load news to process.
      3) Load topic pipeline for this run.
      4) Predict topic ids.
      5) Build topics_per_news dataframe and insert into DB.
    """
    print(f"Running topics inference for id_run={id_run}, mode='{mode}', sources={sources}")

    if mode == "overwrite":
        deleted = delete_existing_assignments(id_run=id_run, sources=sources)
        print(f"Deleted {deleted} previous assignments in topics_per_news.")

    df_news = load_news_to_process(id_run=id_run, sources=sources, mode=mode)

    if df_news.empty:
        print("No news to process for the given parameters. Nothing to do.")
        return

    print(f"Loaded {len(df_news)} news to process.")

    # Load the sklearn Pipeline for this run
    pipeline = load_topic_pipeline(id_run)
    print("Loaded topic pipeline from disk.")

    texts = df_news["text"].tolist()
    news_ids = df_news["id_news"].to_numpy()

    # Predict topic ids. DB texts are already cleaned.
    cluster_labels = predict_topics_for_texts(
        texts=texts,
        pipeline=pipeline,
        apply_cleaning=False,
    )

    df_topics_per_news = build_topics_per_news_df(
        id_run=id_run,
        news_ids=news_ids,
        cluster_labels=np.array(cluster_labels),
    )

    print(f"Built {len(df_topics_per_news)} rows for topics_per_news.")

    engine = get_engine()
    insert_topics_per_news_df(df_topics_per_news, engine=engine)

    print("Topics inference for this run completed successfully.")


def main() -> None:
    """
    Entry point for the topics detector inference pipeline.
    """
    # Load environment variables
    load_dotenv(paths.ENV_FILE)

    # Parse command line arguments
    args = parse_args()
    sources = parse_sources_arg(args.sources)
    mode = args.mode
    run_id = args.run_id

    # Get DB engine
    engine = get_engine()

    if run_id is None:
        run_id = get_active_run_id(engine=engine)
        print(f"No run-id provided. Using active run id_run={run_id}.")
    else:
        print(f"Using provided run id_run={run_id}.")

    run_topics_detector_inference_job(
        id_run=run_id,
        sources=sources,
        mode=mode,
    )


if __name__ == "__main__":
    main()
