from __future__ import annotations

from dotenv import load_dotenv

from news_nlp.config import paths
from news_nlp.db.connection import get_engine
from news_nlp.ner_extractor.model import (
    NerModelConfig,
    load_news_for_ner,
    load_spacy_model,
    extract_entities_for_news,
)
from news_nlp.ner_extractor.tables import (
    build_entities_dataframe,
    build_entities_per_news_dataframe,
)
from news_nlp.ner_extractor.db_io import (
    insert_entities,
    get_entity_mapping,
    save_entities_per_news_dataframe,
)


def main() -> None:
    """
    Run the NER extraction pipeline.

    This pipeline is designed for a "first pass" over the whole corpus.
    For incremental updates with new news, some upsert logic would be required.
    """
    # 1) Load environment variables from .env at project root
    load_dotenv(paths.ENV_FILE)

    # 2) Create engine and load news
    engine = get_engine()
    df_news = load_news_for_ner(engine=engine)
    print(f"Loaded {len(df_news)} news from the database for NER.")

    # 3) Load NER model
    config = NerModelConfig(
        entity_types_to_keep=['PERSON', 'ORG', 'GPE', 'LOC'],
    )
    nlp = load_spacy_model(config)
    print(f"Loaded spaCy model '{config.spacy_model_name}'.")

    # 4) Extract entities
    df_mentions = extract_entities_for_news(df_news, nlp, config)
    print(f"Extracted {len(df_mentions)} entity mentions.")
    if df_mentions.empty:
        print("No entity mentions found. Nothing to do.")
        return
    
    # 5) Build entities dataframe and insert into the database
    df_entities = build_entities_dataframe(df_mentions)
    print(f"Built {len(df_entities)} unique entities.")
    insert_entities(df_entities, engine=engine)    

    # 6) Build mapping (entity_text_norm, entity_type) -> id_entity from the database
    entity_mapping = get_entity_mapping(engine=engine)
    print(f"Entity mapping contains {len(entity_mapping)} entries.")
    if not entity_mapping:
        print("Entity mapping is empty. Nothing to insert into entities_per_news.")
        return

    # 7) Build entities_per_news dataframe using the mapping
    df_entities_per_news = build_entities_per_news_dataframe(df_mentions, entity_mapping)
    print(f"Built {len(df_entities_per_news)} rows for entities_per_news.")

    # 8) Insert entities_per_news into the database
    save_entities_per_news_dataframe(df_entities_per_news, engine=engine)

    print("NER extraction pipeline completed successfully.")


if __name__ == "__main__":
    main()
