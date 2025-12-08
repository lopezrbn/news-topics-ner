from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


# Adjust these paths to your server layout if needed
PROJECT_ROOT = "/home/ubuntu/news-topics-ner"
VENV_PYTHON = f"{PROJECT_ROOT}/.venv/bin/python"

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}


with DAG(
    dag_id="news_topics_ner_initial_setup",
    description="Initial setup for news-topics-ner: load news, train topics model, full inference (train+test).",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # manual trigger
    catchup=False,
    tags=["news-topics-ner", "initial-setup"],
) as dag:

    # 1) Load initial news (train + test) into DB
    load_initial_news = BashOperator(
        task_id="load_initial_news",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{VENV_PYTHON} src/news_nlp/pipelines/01_load_initial_news_pipeline.py"
        ),
    )

    # 2) Train topics detector (unsupervised model + topics tables + MLflow)
    train_topics_detector = BashOperator(
        task_id="train_topics_detector",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{VENV_PYTHON} src/news_nlp/pipelines/02_topics_detector_train_pipeline.py"
        ),
    )

    # 3) Full inference for train + test (topics + entities)
    #    Adjust the CLI args to match your 05_full_inference_pipeline interface.
    full_inference_initial = BashOperator(
        task_id="full_inference_initial",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{VENV_PYTHON} src/news_nlp/pipelines/05_full_inference_pipeline.py "
            f"--mode-topics-detector overwrite --mode-ner-extractor incremental --sources train,test"
        ),
    )

    # Dependencies: 1 -> 2 -> 3
    load_initial_news >> train_topics_detector >> full_inference_initial
