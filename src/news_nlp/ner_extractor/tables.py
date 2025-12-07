from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _normalize_entity_text(text: str) -> str:
    """
    Normalize entity text to group different surface forms of the same entity.

    For now we use a very simple heuristic:
      - strip leading/trailing whitespace
      - convert to lowercase

    This can be improved later (e.g. removing punctuation, handling plurals, etc.).
    """
    return text.strip().lower()


def build_entities_df(
    df_mentions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a dataframe for the entities table, aggregating over the whole corpus.

    Input
    -----
    df_mentions : DataFrame
        Must contain columns:
          - id_news
          - entity_text
          - entity_type

    Output
    ------
    df_entities : DataFrame
        Columns:
          - entity_text         (representative text form; most frequent surface form)
          - entity_text_norm
          - entity_type
          - entity_mentions_count
          - news_count
    """

    cols = [
        "entity_text",
        "entity_text_norm",
        "entity_type",
        "entity_mentions_count",
        "news_count"
    ]

    if df_mentions.empty:
        return pd.DataFrame(
            columns=cols
        )

    df = df_mentions.copy()
    df["entity_text_norm"] = df["entity_text"].astype(str).apply(_normalize_entity_text)

    # Helper function to get the most frequent surface form within each group
    def _most_frequent(series: pd.Series) -> str:
        return series.value_counts().index[0]

    df_entities = (
        df
        .groupby(["entity_text_norm", "entity_type"], as_index=False)
        .agg(
            entity_text=("entity_text", _most_frequent),        # Most frequent surface form
            entity_mentions_count=("id_news", "size"),          # Total number of mentions
            news_count=("id_news", "nunique"),                  # Number of distinct news
        )
    )

    # Reorder columns to keep the same order as in the table definition
    df_entities = df_entities[cols]

    return df_entities


def build_entities_per_news_df(
    df_mentions: pd.DataFrame,
    entity_mapping: Dict[Tuple[str, str], int],
) -> pd.DataFrame:
    """
    Build a dataframe for the entities_per_news table.

    This function uses the raw mentions and an entity_mapping
    (entity_text_norm, entity_type) -> id_entity to produce the final dataframe.

    Input
    -----
    df_mentions : DataFrame
        Must contain columns:
          - id_news
          - entity_text
          - entity_type

    entity_mapping : dict[(entity_text_norm, entity_type) -> id_entity]
        Mapping built from the entities table in the database.

    Output
    ------
    df_entities_per_news : DataFrame
        Columns:
          - id_news
          - id_entity
          - entity_mentions_count
    """
    if df_mentions.empty:
        return pd.DataFrame(columns=["id_news", "id_entity", "entity_mentions_count"])

    df = df_mentions.copy()
    df["entity_text_norm"] = df["entity_text"].astype(str).apply(_normalize_entity_text)

    # Attach id_entity using the mapping
    def _lookup_id_entity(row) -> int | None:
        key = (row["entity_text_norm"], row["entity_type"])
        return entity_mapping.get(key)

    df["id_entity"] = df.apply(_lookup_id_entity, axis=1)

    # Check for missing mappings
    if df["id_entity"].isna().any():
        missing = (
            df[df["id_entity"].isna()][["entity_text_norm", "entity_type"]]
            .drop_duplicates()
        )
        raise ValueError(
            "Some entities in df_mentions do not have an id_entity in the mapping. "
            f"Missing examples:\n{missing.head()}"
        )

    # Aggregate by (id_news, id_entity) to count mentions per news & entity
    grouped = (
        df.groupby(["id_news", "id_entity"], as_index=False)
        .agg(entity_mentions_count=("entity_text", "size"))
    )

    return grouped