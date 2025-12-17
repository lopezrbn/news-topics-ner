from __future__ import annotations

import os
from pathlib import Path

import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from news_nlp.config import paths
from news_nlp.ingestion.simulated_ingestion import load_fraction_prod_into_news_table


DAG_ID = "03_news_topics_ner_daily_ingestion_and_inference"

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

# Must align with DAG start_date (same instant in time)
SIM_START = pendulum.datetime(2025, 12, 17, 20, 30, tz="UTC")
START_LOGICAL_DT_ISO = SIM_START.to_iso8601_string()

SCHEDULE = "*/10 * * * *"  # Every 10 minutes
PERIOD_SECONDS = 600
FRACTION_PER_RUN = 0.01  # 1% of TSV per run

# Environment variable overrides (useful for testing / different setups)
DIR_BASE = Path(os.getenv("NEWS_NLP_DIR_BASE", Path(__file__).parents[1].resolve()))
VENV_PYTHON = os.getenv("NEWS_NLP_VENV_PYTHON", str(DIR_BASE / ".venv/bin/python"))


def ingest_task(
    tsv_path: Path,
    start_logical_dt_iso: str,
    period_seconds: int,
    fraction_per_run: float,
    **context,
) -> int:
    logical_date = context["logical_date"]
    return load_fraction_prod_into_news_table(
        tsv_path=tsv_path,
        start_logical_dt_iso=start_logical_dt_iso,
        period_seconds=period_seconds,
        fraction_per_run=fraction_per_run,
        logical_date=logical_date,
        data_sep="\t",
        loop=True,
    )


with DAG(
    dag_id=DAG_ID,
    description="Simulated ingestion + incremental inference (topics + entities) chained.",
    default_args=DEFAULT_ARGS,
    start_date=SIM_START,
    schedule=SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["news-topics-ner", "simulation", "ingestion", "inference"],
) as dag:

    ingest_simulated_news = PythonOperator(
        task_id="ingest_simulated_news_prod",
        python_callable=ingest_task,
        op_kwargs={
            "tsv_path": paths.DF_TEST_RAW,
            "start_logical_dt_iso": START_LOGICAL_DT_ISO,
            "period_seconds": PERIOD_SECONDS,
            "fraction_per_run": FRACTION_PER_RUN,
        },
        retries=0,
    )

    full_inference_incremental = BashOperator(
        task_id="full_inference_incremental",
        bash_command=(
            f"cd {DIR_BASE} && "
            f"{VENV_PYTHON} src/news_nlp/pipelines/05_full_inference_pipeline.py "
            f"--mode-topics-detector incremental "
            f"--mode-ner-extractor incremental "
            f"--sources prod"
        ),
    )

    ingest_simulated_news >> full_inference_incremental
