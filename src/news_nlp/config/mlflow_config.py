from __future__ import annotations

import os

import mlflow


def init_mlflow() -> None:
    """
    Initialize MLflow tracking for this project.

    It sets the tracking URI and the experiment name using environment variables.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "news-topics-ner")

    if tracking_uri is None:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. Please define it in your .env file."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow initialized with tracking URI={tracking_uri}, experiment={experiment_name}")