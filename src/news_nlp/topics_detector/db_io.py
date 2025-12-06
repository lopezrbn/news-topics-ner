from pathlib import Path
import sys
BASE_DIR = str(Path(__file__).resolve().parents[1])
if BASE_DIR not in sys.path:
    print(f"Adding {BASE_DIR} to sys.path")
    sys.path.insert(0, BASE_DIR)

from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from db.connection import get_engine


def insert_topics_model_training_run_into_db(
    df_run: pd.DataFrame,
    engine: Optional[Engine] = None,
) -> int:
    """
    Insert a single row into topics_model_training_runs table and return the generated id_run.

    df_run must have exactly one row and the columns:
      - id_mlflow_run
      - model_name
      - tfidf_max_features
      - tfidf_max_df
      - tfidf_min_df
      - tfidf_ngram_range
      - tfidf_stop_words
      - svd_n_components
      - kmeans_n_clusters
      - kmeans_n_init
      - top_terms_per_topic
      - random_state
      - silhouette
      - is_active
    """
    if len(df_run) != 1:
        raise ValueError("df_run must contain exactly one row.")

    if engine is None:
        engine = get_engine()

    row = df_run.iloc[0].to_dict()

    query = text(
        """
        INSERT INTO topics_model_training_runs (
            id_mlflow_run,
            model_name,
            tfidf_max_features,
            tfidf_max_df,
            tfidf_min_df,
            tfidf_ngram_range,
            tfidf_stop_words,
            svd_n_components,
            kmeans_n_clusters,
            kmeans_n_init,
            top_terms_per_topic,
            random_state,
            silhouette,
            is_active
        )
        VALUES (
            :id_mlflow_run,
            :model_name,
            :tfidf_max_features,
            :tfidf_max_df,
            :tfidf_min_df,
            :tfidf_ngram_range,
            :tfidf_stop_words,
            :svd_n_components,
            :kmeans_n_clusters,
            :kmeans_n_init,
            :top_terms_per_topic,
            :random_state,
            :silhouette,
            :is_active
        )
        RETURNING id_run
        """
    )

    with engine.begin() as conn:
        result = conn.execute(query, row)
        id_run = result.scalar_one()

    return int(id_run)


def save_topics_dataframe(
    df_topics: pd.DataFrame,
    engine: Optional[Engine] = None,
) -> None:
    """
    Insert rows into topics table using pandas.to_sql.

    df_topics must have columns:
      - id_run
      - id_topic
      - topic_name
      - topic_size
    """
    if engine is None:
        engine = get_engine()

    if df_topics.empty:
        print("No topics to insert.")
        return

    df_topics.to_sql(
        "topics",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1_000,
        method="multi",
    )

    print(f"Inserted {len(df_topics)} rows into 'topics'.")


def save_terms_per_topic_dataframe(
    df_terms_per_topic: pd.DataFrame,
    engine: Optional[Engine] = None,
) -> None:
    """
    Insert rows into terms_per_topic table using pandas.to_sql.

    df_terms_per_topic must have columns:
      - id_run
      - id_topic
      - rank
      - term
      - weight
    """
    if engine is None:
        engine = get_engine()

    if df_terms_per_topic.empty:
        print("No terms_per_topic rows to insert.")
        return

    df_terms_per_topic.to_sql(
        "terms_per_topic",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1_000,
        method="multi",
    )

    print(f"Inserted {len(df_terms_per_topic)} rows into 'terms_per_topic'.")


def save_topics_per_news_dataframe(
    df_topics_per_news: pd.DataFrame,
    engine: Optional[Engine] = None,
) -> None:
    """
    Insert rows into topics_per_news table using pandas.to_sql.

    df_topics_per_news must have columns:
      - id_news
      - id_run
      - id_topic
    """
    if engine is None:
        engine = get_engine()

    if df_topics_per_news.empty:
        print("No topics_per_news rows to insert.")
        return

    df_topics_per_news.to_sql(
        "topics_per_news",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1_000,
        method="multi",
    )

    print(f"Inserted {len(df_topics_per_news)} rows into 'topics_per_news'.")
