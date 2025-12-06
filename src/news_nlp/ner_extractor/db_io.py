from __future__ import annotations

from pathlib import Path
import sys
BASE_DIR = str(Path(__file__).resolve().parents[2])
if BASE_DIR not in sys.path:
    print(f"Adding {BASE_DIR} to sys.path")
    sys.path.insert(0, BASE_DIR)

from typing import Dict, Optional, Tuple

import pandas as pd
from sqlalchemy.engine import Engine

from news_nlp.db.connection import get_engine


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


def save_entities_per_news_dataframe(
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
