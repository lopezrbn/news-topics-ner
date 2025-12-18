# Troubleshooting

This document collects the most common issues when running **news-topics-ner** from a clean clone using **Docker Compose**, plus concrete diagnostic steps and fixes.

---

## 0) Quick diagnosis checklist

When something fails, run these first:

```bash
docker compose ps
docker compose logs -f --tail=200 db
docker compose logs -f --tail=200 api
docker compose logs -f --tail=200 mlflow
docker compose logs -f --tail=200 airflow-webserver
docker compose logs -f --tail=200 airflow-scheduler
```

Confirm:
- containers are **healthy / running**
- host ports are not in use
- `.env` values match the running configuration

---

## 1) Ports already in use

### Symptom
`bind: address already in use` when starting Docker Compose.

### Fix
1. Identify what is using the port:
   ```bash
   sudo lsof -i :8080
   sudo lsof -i :8000
   sudo lsof -i :5000
   sudo lsof -i :5432
   ```

2. Change host ports in `.env` (e.g. `AIRFLOW_PORT_HOST`, `API_PORT_HOST`, etc.), then restart:
   ```bash
   docker compose down
   docker compose up --build
   ```

---

## 2) Postgres container fails to start / init scripts not applied

### Symptom
- DB container restarts repeatedly
- Airflow/MLflow/API cannot connect
- `psql` fails, or databases/users are missing

### Diagnostics
```bash
docker compose logs -f db
```

### Fix
If you want a clean DB re-init:

```bash
docker compose down -v
docker compose up --build
```

Note: `-v` deletes the DB volume, so init SQL will run again.

---

## 3) Airflow UI not accessible / Airflow containers crash

### Symptom
- `http://localhost:8080` does not load
- Airflow containers exit with errors

### Diagnostics
```bash
docker compose logs -f airflow-webserver
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-init
```

### Common causes and fixes

#### 3.1 Missing/invalid Fernet key

Symptom in logs:
- errors mentioning `FERNET_KEY` or cryptography

Fix:
- Ensure `.env` contains `AIRFLOW_FERNET_KEY`.
- Generate a valid key:
  ```bash
  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  ```

Restart:
```bash
docker compose down
docker compose up --build
```

#### 3.2 Airflow DB not initialized

Fix:
- reset volumes and rerun init:
  ```bash
  docker compose down -v
  docker compose up --build
  ```

---

## 4) DAGs do not appear in Airflow

### Symptom
- Airflow UI loads but no project DAGs show up.

### Diagnostics
1. Check the DAG files exist:
   ```bash
   ls -la airflow_dags/
   ```

2. Check Airflow sees the mounted DAG folder:
   ```bash
   docker compose exec airflow-webserver ls -la /opt/airflow/dags
   ```

3. Check scheduler logs for import errors:
   ```bash
   docker compose logs -f airflow-scheduler
   ```

### Common causes and fixes

#### 4.1 Import errors (PYTHONPATH)

Symptom:
- stack traces during DAG import, often `ModuleNotFoundError: news_nlp`

Fix:
- Ensure the Airflow service sets:
  - `PYTHONPATH=/opt/airflow/news-topics-ner/src`
- This is configured in `docker-compose.yml`. If you changed compose mounts/env vars, revert to repo defaults.

Restart:
```bash
docker compose down
docker compose up --build
```

#### 4.2 Syntax error in a DAG file

Fix:
- Open the failing DAG, fix the syntax error, and restart scheduler:
  ```bash
  docker compose restart airflow-scheduler
  ```

---

## 5) MLflow UI not accessible / MLflow server errors

### Symptom
- `http://localhost:5000` does not load
- MLflow container exits or logs DB connection errors

### Diagnostics
```bash
docker compose logs -f mlflow
```

### Fix
- Ensure Postgres is up and `mlflow_db` exists.
- Reset volumes if needed:
  ```bash
  docker compose down -v
  docker compose up --build
  ```

---

## 6) API errors (FastAPI)

### 6.1 `GET /health` fails

Diagnostics:
```bash
docker compose logs -f api
docker compose exec api python -c "import news_nlp; print('ok')"
```

Fix:
- Ensure API container is running and port mapping matches `.env`.
- Restart:
  ```bash
  docker compose restart api
  ```

---

### 6.2 `FileNotFoundError` when loading topic model artifacts

### Symptom
Calling `/v1/topics` or `/v1/analyze` fails with an error indicating missing artifacts.

### Root cause
The repo does not ship trained artifacts in Git; they are generated at runtime under `models/`.

### Fix
Run training + inference via Airflow:
1. Start stack
2. Trigger DAG:
   - `01_news_topics_ner_initial_setup`

Or run manually inside the container:
```bash
docker compose exec api python src/news_nlp/pipelines/02_topics_detector_train_pipeline.py
docker compose exec api python src/news_nlp/pipelines/05_full_inference_pipeline.py   --mode-topics-detector overwrite   --mode-ner-extractor incremental   --sources train,test
```

---

### 6.3 “No active run” (topics run resolution fails)

### Symptom
API cannot resolve a topics run id (no active run in DB).

### Fix
- Train the topics detector (creates and activates a run):
  ```bash
  docker compose exec api python src/news_nlp/pipelines/02_topics_detector_train_pipeline.py
  ```

---

### 6.4 `422 Unprocessable Entity`

### Symptom
Request body does not match the Pydantic schema.

### Fix
- Validate the payload against Swagger at:
  - `http://localhost:8000/docs`

---

## 7) Training fails because OpenAI key is missing/invalid

### Symptom
Training pipeline fails during topic naming / prompt execution.

### Fix options
Option A (recommended for full features):
- Set `OPENAI_API_KEY` in `.env`, restart and rerun training.

Option B (run without LLM naming):
- If your pipeline supports skipping naming when the key is not set, keep the key unset and rerun training.
- In that case, topics remain interpretable via top terms stored in `terms_per_topic`.

---

## 8) Ingestion fails / dataset not found

### Symptom
`01_load_initial_news_pipeline.py` fails because `data/raw/train.tsv` or `data/raw/test.tsv` is missing.

### Root cause
The raw TSVs are extracted from:
- `data/compressed/data.zip`

### Fix
- Ensure `data/compressed/data.zip` exists in the repo.
- Re-run ingestion:
  ```bash
  docker compose exec api python src/news_nlp/pipelines/01_load_initial_news_pipeline.py
  ```

If the zip is missing, re-download the repository from GitHub.

---

## 9) Duplicate rows in prod ingestion simulation

### Symptom
The `prod` source contains duplicates after repeated runs.

### Likely causes
- ingestion task retries
- schedule backfills (Airflow catchup)
- manual re-runs without deduplication constraints

### Fix
- Ensure `catchup=False` in the ingestion DAG if you want to avoid backfills.
- Keep `retries=0` for the ingestion task to prevent duplicates on retry.
- If duplicates already exist, clean the DB state:
  ```bash
  docker compose down -v
  docker compose up --build
  ```

---

## 10) Resetting to a clean state

If you want to recreate the environment exactly as a reviewer would from a clean clone:

```bash
docker compose down -v
docker compose up --build
```

This resets:
- Postgres databases (business + airflow + mlflow)
- MLflow artifact store

Then trigger:
- `01_news_topics_ner_initial_setup`

---

## Related documents

- `docs/SETUP_AND_DEPLOYMENT.md`
- `docs/PIPELINES_AND_DAGS.md`
- `docs/API_REFERENCE.md`
- `docs/ARCHITECTURE.md`
