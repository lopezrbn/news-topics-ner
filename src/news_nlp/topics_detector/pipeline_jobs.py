from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from news_nlp.db.connection import get_engine
from news_nlp.topics_detector.model import load_topic_pipeline
from news_nlp.topics_detector.inference import predict_topics_for_texts
from news_nlp.topics_detector.tables import build_topics_per_news_df
from news_nlp.topics_detector.db_io import (
    delete_existing_assignments,
    load_news_to_process,
    insert_topics_per_news_df,
)


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