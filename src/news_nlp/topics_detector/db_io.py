from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from news_nlp.db.connection import get_engine


def get_active_run_id(engine: Engine | None = None) -> int:
    """
    Return the id_run of the active model in the "topics_model_training_run" table.

    If no active run is found, raises a ValueError.

    The active run is defined as the row in topics_model_training_runs
    with is_active = true. The partial unique index on is_active=true
    should guarantee at most one such row.
    """
    if engine is None:
        engine = get_engine()

    sql = text(
        """
        SELECT id_run
        FROM topics_model_training_runs
        WHERE is_active = true
        ORDER BY created_at DESC
        LIMIT 1
        """
    )

    with engine.connect() as conn:
        result = conn.execute(sql).scalar()

    if result is None:
        raise ValueError("No active topics_model_training_runs found (is_active = true).")

    return int(result)


def load_training_texts() -> pd.Series:
    """
    Load training texts (source='train') from the `news` table.

    Returns
    -------
    texts : pandas Series
        Series of cleaned texts to be used for training.
    """
    engine = get_engine()

    query = """
        SELECT text
        FROM news
        WHERE source = 'train'
          AND text IS NOT NULL
    """

    df = pd.read_sql(query, con=engine)
    if df.empty:
        raise ValueError("No training texts found in news (source='train').")

    return df["text"]


def load_news_without_topics_for_run(id_run: int) -> pd.DataFrame:
    """
    Load news that do not yet have a topic assigned for the given run.

    Returns a dataframe with columns:
      - id_news
      - text
    """
    engine = get_engine()

    query = """
        SELECT n.id_news, n.text
        FROM news AS n
        LEFT JOIN topics_per_news AS t
          ON n.id_news = t.id_news
         AND t.id_run = %(id_run)s
        WHERE t.id_news IS NULL
          AND n.text IS NOT NULL
    """

    df = pd.read_sql(query, con=engine, params={"id_run": id_run})
    return df


def insert_topics_model_training_run_df(
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


def insert_topics_df(
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


def insert_terms_per_topic_df(
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


def insert_topics_per_news_df(
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


def delete_existing_assignments(
    id_run: int,
    sources: Optional[Iterable[str]],
) -> int:
    """
    Delete existing rows in topics_per_news for a given run and optional sources.

    Parameters
    ----------
    id_run : int
        Run identifier in topics_model_training_runs.
    sources : iterable of str or None
        If provided, only rows whose news.source is in this list are deleted.
        If None, rows for all sources are deleted for this run.

    Returns
    -------
    deleted_count : int
        Number of deleted rows (as reported by the DB).
    """
    engine = get_engine()

    if sources is None:
        # Delete all assignments for this run_id
        sql = text(
            """
            DELETE FROM topics_per_news
            WHERE id_run = :id_run
            """
        )
        params = {"id_run": id_run}
    else:
        # Delete assignments only for news in the specified sources
        sql = text(
            """
            DELETE FROM topics_per_news
            WHERE id_run = :id_run
                AND id_news IN (
                    SELECT id_news
                    FROM news
                    WHERE source = ANY(:sources)
                )
            """
        )
        params = {"id_run": id_run, "sources": list(sources)}

    with engine.begin() as conn:
        result = conn.execute(sql, params)
        deleted_count = result.rowcount if result.rowcount is not None else -1

    return deleted_count


def load_news_to_process(
    id_run: int,
    sources: Optional[Iterable[str]],
    mode: str,
) -> pd.DataFrame:
    """
    Load news that should be processed for topics inference.

    Parameters
    ----------
    id_run : int
        Run identifier in topics_model_training_runs.
    sources : iterable of str or None
        If provided, only news with these sources are considered.
        If None, all sources are included.
    mode : {"incremental", "overwrite"}
        - "incremental": only news that do not yet have a row in topics_per_news
          for this run are loaded.
        - "overwrite": all news matching the sources are loaded, assuming any
          previous assignments have already been deleted.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns:
          - id_news
          - text
    """
    engine = get_engine()

    if mode not in ("incremental", "overwrite"):
        raise ValueError(f"Invalid mode: {mode}. Use 'incremental' or 'overwrite'.")

    # Base query: select news + left join with topics_per_news for this run
    # We will filter by source if needed, and by whether topics_per_news exists.
    base_query = """
        SELECT n.id_news, n.text
        FROM news AS n
        LEFT JOIN topics_per_news AS t
            ON n.id_news = t.id_news
            AND t.id_run = :id_run
    """

    conditions = []
    params: dict = {"id_run": id_run}

    if sources is not None:
        conditions.append("n.source = ANY(:sources)")
        params["sources"] = list(sources)

    if mode == "incremental":
        # Only news without topics_per_news row for this run
        conditions.append("t.id_news IS NULL")
    else:
        # overwrite: assume we have already deleted rows for this run & sources
        # -> all news matching this source filter are candidates
        pass

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    df = pd.read_sql(base_query, con=engine, params=params)
    return df