from dotenv import load_dotenv

from news_nlp.config import paths
from news_nlp.db.connection import get_engine
from news_nlp.topics_detector.model import (
    TopicModelConfig,
    TopicModelArtifacts,
    train_topic_model,
    compute_top_terms_per_topic,
    save_topic_model_artifacts,
)
from news_nlp.topics_detector.tables import (
    build_topics_model_training_run_df,
    build_topics_df,
    build_terms_per_topic_df,
)
from news_nlp.topics_detector.db_io import (
    load_training_texts,
    insert_topics_model_training_run_df,
    insert_topics_df,
    insert_terms_per_topic_df,
)
from news_nlp.topics_detector.topics_naming import (
    generate_topic_names_with_llm
)


def main() -> None:
    """
    Topics detector training pipeline.

    Steps:
      1) Load environment variables and DB connection
      2) Load training texts from the database
      3) Train topic model (sklearn Pipeline)
      4) Insert training run metadata into topics_model_training_runs
      5) Build and insert topics and terms_per_topic tables
      6) Save model artifacts to disk
    """
    # 1) Load environment variables and DB connection
    load_dotenv(paths.ENV_FILE)
    engine = get_engine()

    # 2) Load training texts
    texts = load_training_texts()
    print(f"Loaded {len(texts)} training texts from news (source='train').")

    # 3) Train topic model
    input_config = TopicModelConfig()
    output_artifacts: TopicModelArtifacts = train_topic_model(
        texts=texts.tolist(),
        config=input_config,
    )
    print(f"Model trained. Silhouette score: {output_artifacts.silhouette:.4f}")

    # 4) Insert training run metadata into "topics_model_training_runs" table
    df_run = build_topics_model_training_run_df(input_config, output_artifacts)
    id_run = insert_topics_model_training_run_df(df_run, engine=engine)
    print(f"Inserted topics_model_training_run with id_run={id_run}")

    # 5) Build and insert topics and terms_per_topic

    # 5.1) Compute top terms per topic (using your existing logic)
    top_terms_per_topic = compute_top_terms_per_topic(input_config, output_artifacts)

    # 5.2) Generate topic names with LLM
    topic_names = generate_topic_names_with_llm(top_terms_per_topic)

    # 5.3) Build dataframe for "topics" table
    df_topics = build_topics_df(
        id_run=id_run,
        topic_names=topic_names,
        artifacts=output_artifacts,
    )

    # 5.4) Build dataframe for "terms_per_topic" table
    df_terms_per_topic = build_terms_per_topic_df(
        id_run=id_run,
        top_terms_per_topic=top_terms_per_topic,
    )

    # 5.5) Insert tables into the database
    insert_topics_df(df_topics, engine=engine)
    insert_terms_per_topic_df(df_terms_per_topic, engine=engine)

    # 6) Save model artifacts to disk (pipeline + components)
    save_topic_model_artifacts(id_run, output_artifacts)

    print("Topics detector training pipeline (train only) completed successfully.")


if __name__ == "__main__":
    main()
