from __future__ import annotations

import os
from pathlib import Path

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


# Environment variable overrides (useful for testing / different setups)
DIR_BASE = Path(
    os.getenv("NEWS_NLP_DIR_BASE", Path(__file__).parents[1].resolve())
)
VENV_PYTHON = os.getenv(
    "NEWS_NLP_VENV_PYTHON", DIR_BASE / ".venv/bin/python"
)

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}


with DAG(
    dag_id="02_news_topics_ner_retrain",
    description=(
        "Retrain topics detector model and optionally recompute topics + entities "
        "for selected sources."
    ),
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # manual trigger
    catchup=False,
    tags=["news-topics-ner", "retrain"],
) as dag:

    # 1) Retrain topics detector
    train_topics_detector = BashOperator(
        task_id="train_topics_detector",
        bash_command=(
            f"cd {DIR_BASE} && "
            f"{VENV_PYTHON} src/news_nlp/pipelines/02_topics_detector_train_pipeline.py"
        ),
    )

    # 2) Recompute full inference for prod (overwrite) using the new active run
    full_inference_after_retrain = BashOperator(
        task_id="full_inference_after_retrain",
        bash_command=(
            f"cd {DIR_BASE} && "
            f"{VENV_PYTHON} src/news_nlp/pipelines/05_full_inference_pipeline.py "
            f"--mode-topics-detector overwrite --mode-ner-extractor incremental --sources all"
        ),
    )

    train_topics_detector >> full_inference_after_retrain
