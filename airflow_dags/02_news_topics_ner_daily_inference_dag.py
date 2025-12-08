from __future__ import annotations

from pathlib import Path

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


DIR_BASE = Path(__file__).parents[1].resolve()
VENV_PYTHON = DIR_BASE / ".venv/bin/python"

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}


with DAG(
    dag_id="news_topics_ner_daily_inference",
    description="Daily incremental inference for new prod news (topics + entities).",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    # schedule_interval="0 7 * * *",  # every day at 06:00
    schedule_interval=None,            # Manual launching for the moment as we don't have production news feed
    catchup=False,
    tags=["news-topics-ner", "inference", "daily"],
) as dag:

    full_inference_incremental = BashOperator(
        task_id="full_inference_incremental",
        bash_command=(
            f"cd {DIR_BASE} && "
            f"{VENV_PYTHON} src/news_nlp/pipelines/05_full_inference_pipeline.py "
            f"--mode-topics-detector incremental --mode-ner-extractor incremental --sources prod"
        ),
    )
