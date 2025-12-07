from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

from news_nlp.config import paths
from news_nlp.db.connection import get_engine
from news_nlp.ner_extractor.model import (
    NerModelConfig,
    load_spacy_model,
    extract_entities_for_news,
)
from news_nlp.ner_extractor.db_io import (
    load_news_to_process,
    insert_entities,
    get_entity_mapping,
    save_entities_per_news_df,
)
from news_nlp.ner_extractor.tables import (
    build_entities_df,
    build_entities_per_news_df,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the NER inference pipeline.
    """
    parser = argparse.ArgumentParser(
        description="NER extractor inference pipeline (incremental, parametrizable by sources)."
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

    # Only incremental mode is supported for NER, to keep things simple for now
    parser.add_argument(
        "--mode",
        type=str,
        choices=["incremental"],
        default="incremental",
        help=(
            "Inference mode. "
            "Currently only 'incremental' is supported for NER: "
            "process news that do not yet have entities_per_news rows."
        ),
    )

    return parser.parse_args()


def parse_sources_arg(sources: str) -> Optional[List[str]]:
    """
    Parse a comma-separated sources string.

    Examples
    --------
    "train"      -> ["train"]
    "train, test" -> ["train", "test"]
    "all"        -> None  (meaning: no filtering by source)

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


def run_ner_inference_job(
    sources: Optional[List[str]],
) -> None:
    """
    Run incremental NER inference for the selected sources.

    Steps:
      1) Load news without entities_per_news for the given sources.
      2) Load spaCy model.
      3) Extract entity mentions for those news.
      4) Build entities and entities_per_news dataframes.
      5) Insert/update entities and insert entities_per_news into the DB.
    """
    print(f"Running NER inference job for sources={sources} (incremental mode).")

    engine = get_engine()

    # 1) Load news to process (those without entities_per_news)
    df_news = load_news_to_process(sources=sources)
    if df_news.empty:
        print("No news to process for NER (all have entities already). Nothing to do.")
        return
    print(f"Loaded {len(df_news)} news to process for NER.")

    # 2) Load spaCy model
    ner_config = NerModelConfig()
    nlp = load_spacy_model(ner_config)
    print(f"Loaded spaCy model '{ner_config.spacy_model_name}'")

    # 3) Extract entity mentions for these news
    #    This function should return a dataframe with one row per entity mention
    df_mentions = extract_entities_for_news(
        df_news=df_news,
        nlp=nlp,
        config=ner_config,
    )
    if df_mentions.empty:
        print("No entities found in the selected news.")
        return
    print(f"Extracted {len(df_mentions)} entity mentions from {len(df_news)} news.")

    # 4) Build entities dataframe
    df_entities = build_entities_df(df_mentions)
    print(f"Built {len(df_entities)} unique entities.")

    # 5) Insert entities into the database
    insert_entities(df_entities, engine=engine)

    # 6) Build mapping (entity_text_norm, entity_type) -> id_entity from the database
    entity_mapping = get_entity_mapping(engine=engine)
    print(f"Got mapping for {len(entity_mapping)} entities from DB.")

    # 7) Build entities_per_news dataframe using the mapping
    df_entities_per_news = build_entities_per_news_df(
        df_mentions=df_mentions,
        entity_mapping=entity_mapping,
    )
    print(f"Built entities_per_news dataframe with {len(df_entities_per_news)} rows.")

    # 8) Insert entities_per_news into the DB
    save_entities_per_news_df(df_entities_per_news, engine=engine)

    print("NER inference job completed successfully.")


def main() -> None:
    """
    Entry point for the NER extractor inference pipeline.
    """
    # 1) Load environment variables
    load_dotenv(paths.ENV_FILE)

    # 2) Parse CLI arguments
    args = parse_args()
    sources = parse_sources_arg(args.sources)
    mode = args.mode  # currently only "incremental" is accepted
    print(f"Mode: {mode} (only 'incremental' is implemented for NER).")

    # 3) Run NER inference job
    run_ner_inference_job(sources=sources)


if __name__ == "__main__":
    main()
