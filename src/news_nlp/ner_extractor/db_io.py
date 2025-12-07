from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, Iterable

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from news_nlp.db.connection import get_engine


# def load_news_for_ner(
#     engine: Optional[Engine] = None,
# ) -> pd.DataFrame:
#     """
#     Load news from the database to run NER on.

#     As the model is pretrained, we don't need train/test split here, so we just
#     load all news with non-null text.

#     Returns
#     -------
#     df_news : DataFrame
#         Dataframe with columns:
#           - id_news
#           - text
#     """
#     if engine is None:
#         engine = get_engine()

#     query = """
#         SELECT id_news, text
#         FROM news
#         WHERE text IS NOT NULL
#     """

#     df_news = pd.read_sql(query, con=engine)
#     if df_news.empty:
#         raise ValueError("No news found in the database (text IS NOT NULL).")

#     return df_news
def load_news_to_process(
    sources: Optional[Iterable[str]],
) -> pd.DataFrame:
    """
    Load news that should be processed for NER.

    We consider that a news item needs NER processing if it has no rows
    in entities_per_news yet.

    Parameters
    ----------
    sources : iterable of str or None
        If provided, only news with these sources are considered
        (e.g. 'train', 'test', 'prod').
        If None, all sources are included.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns:
          - id_news
          - text
    """
    engine = get_engine()

    base_query = text("""
        SELECT n.id_news, n.text
        FROM news AS n
        LEFT JOIN entities_per_news AS e
          ON n.id_news = e.id_news
    """)

    conditions = []
    params: Dict[str, Any] = {}

    if sources is not None:
        conditions.append("n.source = ANY(:sources)")
        params["sources"] = list(sources)

    # Only news without entities_per_news rows
    conditions.append("e.id_news IS NULL")

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    df = pd.read_sql(base_query, con=engine, params=params)
    return df


def insert_entities(
    df_entities: pd.DataFrame,
    engine: Optional[Engine] = None,
) -> None:
    """
    Insert rows into the entities table using pandas.to_sql.

    df_entities must have columns:
      - entity_text
      - entity_text_norm
      - entity_type
      - entity_mentions_count
      - news_count
    """
    if engine is None:
        engine = get_engine()

    if df_entities.empty:
        print("No entities to insert.")
        return

    df_entities.to_sql(
        "entities",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1_000,
        method="multi",
    )

    print(f"Inserted {len(df_entities)} rows into 'entities'.")


def get_entity_mapping(
    engine: Optional[Engine] = None,
) -> Dict[Tuple[str, str], int]:
    """
    Build a mapping (entity_text_norm, entity_type) -> id_entity
    from the entities table.

    This is useful after inserting new entities, so we can attach id_entity
    when building entities_per_news.
    """
    if engine is None:
        engine = get_engine()

    query = """
        SELECT id_entity, entity_text_norm, entity_type
        FROM entities
    """
    df_db = pd.read_sql(query, con=engine)

    mapping: Dict[Tuple[str, str], int] = {}
    for _, row in df_db.iterrows():
        key = (str(row["entity_text_norm"]), str(row["entity_type"]))
        mapping[key] = int(row["id_entity"])

    return mapping


def save_entities_per_news_df(
    df_entities_per_news: pd.DataFrame,
    engine: Optional[Engine] = None,
) -> None:
    """
    Insert rows into entities_per_news table using pandas.to_sql.

    df_entities_per_news must have columns:
      - id_news
      - id_entity
      - entity_mentions_count
    """
    if engine is None:
        engine = get_engine()

    if df_entities_per_news.empty:
        print("No entities_per_news rows to insert.")
        return

    df_entities_per_news.to_sql(
        "entities_per_news",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1_000,
        method="multi",
    )

    print(f"Inserted {len(df_entities_per_news)} rows into 'entities_per_news'.")
