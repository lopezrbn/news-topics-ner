from dotenv import load_dotenv

from news_nlp.config import paths
from news_nlp.db.connection import get_engine
from news_nlp.topics_detector.model import (
    TopicModelConfig,
    TopicModelArtifacts,
    load_training_news,
    train_topic_model,
    compute_top_terms_per_topic,
    save_topic_model_artifacts,
)
from news_nlp.topics_detector.tables import (
    build_topics_model_training_run_row,
    build_topics_dataframe,
    build_terms_per_topic_dataframe,
    build_topics_per_news_dataframe,
)
from news_nlp.topics_detector.db_io import (
    insert_topics_model_training_run_into_db,
    save_topics_dataframe,
    save_terms_per_topic_dataframe,
    save_topics_per_news_dataframe,
)
from news_nlp.topics_detector.naming import generate_topic_names_with_llm


def main() -> None:
    """
    Train the topics detector model, persist model training run metadata,
    topics, terms per topic and topics per news into the database,
    and save model artifacts to disk.
    """
    # 0) Load environment variables from .env
    load_dotenv(paths.ENV_FILE)

    # 1) Load data
    engine = get_engine()
    df_train = load_training_news(engine=engine)
    texts = df_train["text"].tolist()
    news_ids = df_train["id_news"].to_numpy()
    print(f"Loaded {len(df_train)} training news from the database.")

    # 2) Train model
    input_config = TopicModelConfig()
    output_artifacts: TopicModelArtifacts = train_topic_model(texts, input_config)
    print(f"Training finished. Silhouette score: {output_artifacts.silhouette:.4f}")

    # 3) Generate row for "topics_model_training_runs" db table and insert it
    df_run = build_topics_model_training_run_row(input_config, output_artifacts)
    id_run = insert_topics_model_training_run_into_db(df_run, engine=engine)
    print(f"Created topics_model_training_runs row with id_run={id_run}")

    # 4) Compute top terms per topic and generate topic names with LLM
    top_terms_per_topic = compute_top_terms_per_topic(input_config, output_artifacts)
    topic_names = generate_topic_names_with_llm(top_terms_per_topic)
    print("Generated topic names using LLM.")

    # 5) Build dataframes for "topics", "terms_per_topic" and "topics_per_news tables"
    df_topics = build_topics_dataframe(id_run, output_artifacts.cluster_labels, topic_names)
    df_terms_per_topic = build_terms_per_topic_dataframe(id_run, top_terms_per_topic)
    df_topics_per_news = build_topics_per_news_dataframe(id_run, news_ids, output_artifacts.cluster_labels)

    # 6) Save tables into the database
    save_topics_dataframe(df_topics, engine=engine)
    save_terms_per_topic_dataframe(df_terms_per_topic, engine=engine)
    save_topics_per_news_dataframe(df_topics_per_news, engine=engine)

    # 7) Save model artifacts to disk
    save_topic_model_artifacts(id_run, output_artifacts)

    print("Topics detector training pipeline completed successfully.")


if __name__ == "__main__":
    main()
