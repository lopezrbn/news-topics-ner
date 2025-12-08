from __future__ import annotations

from dotenv import load_dotenv

import mlflow
from typing import Tuple

from news_nlp.config import paths
from news_nlp.config.mlflow_config import init_mlflow
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
    set_topics_model_training_run_active,
)
from news_nlp.topics_detector.topics_naming import (
    generate_topic_names_with_llm
)


def log_mlflow_params_from_config(config: TopicModelConfig) -> None:
    """
    Log TopicModelConfig parameters into the current MLflow run.
    """
    # General
    mlflow.log_param("model_name", config.model_name)
    mlflow.log_param("top_terms_per_topic", config.top_terms_per_topic)
    mlflow.log_param("random_state", config.random_state)

    # TF-IDF
    mlflow.log_param("tfidf_max_features", config.tfidf_max_features)
    mlflow.log_param("tfidf_max_df", config.tfidf_max_df)
    mlflow.log_param("tfidf_min_df", config.tfidf_min_df)
    mlflow.log_param(
        "tfidf_ngram_range",
        f"{config.tfidf_ngram_range[0]}_{config.tfidf_ngram_range[1]}",
    )
    mlflow.log_param("tfidf_stop_words", config.tfidf_stop_words)

    # Truncated SVD
    mlflow.log_param("svd_n_components", config.svd_n_components)

    # K-Means
    mlflow.log_param("kmeans_n_clusters", config.kmeans_n_clusters)
    mlflow.log_param("kmeans_n_init", str(config.kmeans_n_init))

    # Extra static params
    mlflow.log_param("vectorizer_type", "TfidfVectorizer")
    # mlflow.log_param("svd_solver", "randomized")
    # mlflow.log_param("kmeans_init", "k-means++")


def main() -> None:
    """
    Topics detector training pipeline.

    Steps:
      1) Load environment variables and DB connection
      2) Initialize MLflow
      3) Load training texts from the database
      4) Start MLflow run and train topic model
      5) Insert training run metadata into topics_model_training_runs
      6) Build and insert topics and terms_per_topic tables
      7) Save model artifacts to disk
    """
    # 1) Load environment variables and DB connection
    load_dotenv(paths.ENV_FILE)
    engine = get_engine()

    # 2) Initialize MLflow
    init_mlflow()

    # 3) Load training texts
    texts = load_training_texts()
    print(f"Loaded {len(texts)} training texts from news (source='train').")

    # 4) Start MLflow run and train topic model
    input_config = TopicModelConfig()

    with mlflow.start_run(run_name=input_config.model_name) as mlflow_run:
        id_mlflow_run = mlflow_run.info.run_id
        print(f"Started MLflow run with id_mlflow_run: {id_mlflow_run}")

        # 4.1) Log hyperparameters from config
        log_mlflow_params_from_config(input_config)

        # 4.2) Train topic model
        output_artifacts: TopicModelArtifacts = train_topic_model(
            texts=texts.tolist(),
            config=input_config,
        )
        # Log trained model to MLflow
        model = output_artifacts.pipeline
        mlflow.sklearn.log_model(model, name="topics_detector_model")
        # Log silhouette score to MLflow
        silhouette = output_artifacts.silhouette
        mlflow.log_metric("silhouette_score", silhouette)
        print(f"Model trained. Silhouette score: {silhouette:.4f}")

        # 4.3) Insert training run metadata into "topics_model_training_runs" table
        df_run = build_topics_model_training_run_df(input_config, output_artifacts, id_mlflow_run)
        id_run = insert_topics_model_training_run_df(df_run, engine=engine)
        print(f"Inserted topics_model_training_run with id_run={id_run}")

        # 4.4) Build and insert topics and terms_per_topic

        # 4.4.1) Compute top terms per topic (using your existing logic)
        top_terms_per_topic = compute_top_terms_per_topic(input_config, output_artifacts)

        # 4.4.2) Generate topic names with LLM
        topic_names = generate_topic_names_with_llm(top_terms_per_topic)

        # 4.4.3) Build dataframe for "topics" table
        df_topics = build_topics_df(
            id_run=id_run,
            topic_names=topic_names,
            artifacts=output_artifacts,
        )

        # 4.4.4) Build dataframe for "terms_per_topic" table
        df_terms_per_topic = build_terms_per_topic_df(
            id_run=id_run,
            top_terms_per_topic=top_terms_per_topic,
        )

        # 4.4.5) Insert tables into the database
        insert_topics_df(df_topics, engine=engine)
        insert_terms_per_topic_df(df_terms_per_topic, engine=engine)

        # 4.5) Save model artifacts to disk (pipeline + components) and log them also in MLflow
        artifact_paths = save_topic_model_artifacts(id_run, output_artifacts)
        for _, artifact_path in artifact_paths.items():
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path=="topics_detector_model")

        # 4.6) After everything succeded, mark the run as active
        set_topics_model_training_run_active(id_run, engine)

    print("Topics detector training pipeline (train only) completed successfully.")


if __name__ == "__main__":
    main()
