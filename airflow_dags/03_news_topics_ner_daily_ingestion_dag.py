import pendulum
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from news_nlp.config import paths
from news_nlp.ingestion.simulated_ingestion import load_fraction_prod_into_news_table

DAG_ID = "03_news_topics_ner_daily_ingestion"

# Must align with DAG start_date (same instant in time)
START_LOGICAL_DT_ISO = "2025-12-01T00:00:00+00:00"

# TSV_PATH = Path("/opt/airflow/data/raw/test.tsv")

# SCHEDULE = "0 * * * *"    # Hourly schedule
# PERIOD_SECONDS = 3600
SCHEDULE = "*/1 * * * *"    # Minutely schedule
PERIOD_SECONDS = 60

FRACTION_PER_RUN = 0.01     # 1% of TSV per run

def ingest_task(tsv_path: str, start_logical_dt_iso: str, period_seconds: int, fraction_per_run: float, **context):
    logical_date = context["logical_date"]
    return load_fraction_prod_into_news_table(
        tsv_path=Path(tsv_path),
        start_logical_dt_iso=start_logical_dt_iso,
        period_seconds=period_seconds,
        fraction_per_run=fraction_per_run,
        logical_date=logical_date,
        data_sep="\t",
    )

with DAG(
    dag_id=DAG_ID,
    start_date=pendulum.datetime(2025, 12, 1, tz="UTC"),
    schedule=SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["simulation", "ingestion"],
) as dag:

    ingest_simulated_news = PythonOperator(
        task_id="ingest_simulated_news_prod",
        python_callable=ingest_task,
        op_kwargs={
            "tsv_path": str(paths.DF_TEST_RAW),
            "start_logical_dt_iso": START_LOGICAL_DT_ISO,
            "period_seconds": PERIOD_SECONDS,
            "fraction_per_run": FRACTION_PER_RUN,
        },
        retries=0,  # recommended with current schema to avoid duplicates on retry
    )
