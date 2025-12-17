# Setup and Deployment

This document explains how to run **news-topics-ner** from a clean GitHub clone in a fully reproducible way.

The **official** execution mode is **single-host Docker Compose** (local deployment). A lightweight local (non-Docker) setup is also described for development purposes, but Docker is the recommended path for reviewers.

---

## Prerequisites

### Required
- Docker Engine
- Docker Compose (v2)

### Optional (for local / non-Docker runs)
- Python 3.10
- `make` (optional convenience)
- A local Postgres instance (if you want to run without Docker)

---

## 1) Clone the repository

```bash
git clone <REPO_URL>
cd news-topics-ner
```

---

## 2) Configure environment variables

The Docker setup reads environment variables from a `.env` file at the repository root.

### 2.1 Create `.env` from the example

```bash
cp .env.example .env
```

### 2.2 Set required values

You must set at least:

- `OPENAI_API_KEY`  
  Used by the topics training pipeline to generate topic names with an LLM (prompted via `config/prompts.yaml`).

### 2.3 Airflow keys (required for a clean Airflow boot)

Airflow requires a Fernet key:

- `AIRFLOW_FERNET_KEY`

Generate one with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Paste the output into:

```text
AIRFLOW_FERNET_KEY="..."
```

The example also includes:

- `AIRFLOW_SECRET_KEY`  
  Used by the webserver session security. For this technical project, any long random string is acceptable.

### 2.4 Ports and host conflicts

The default host ports match the container ports:

- Postgres: `DB_PORT_HOST=5432`
- API: `API_PORT_HOST=8000`
- Airflow: `AIRFLOW_PORT_HOST=8080`
- MLflow: `MLFLOW_PORT_HOST=5000`

If any of these ports are already in use on your machine, change the `*_PORT_HOST` values in `.env`.

---

## 3) Start the stack (Docker Compose)

From repository root:

```bash
docker compose up --build
```

This will build the shared Python image (used by the API and the MLflow server) and start all services:
- Postgres (`db`)
- MLflow server (`mlflow`)
- FastAPI service (`api`)
- Airflow init (`airflow-init`), webserver (`airflow-webserver`), scheduler (`airflow-scheduler`)

### Run in background (optional)

```bash
docker compose up -d --build
```

---

## 4) Verify services

### 4.1 Check running containers

```bash
docker compose ps
```

### 4.2 Health endpoints

- API health:
  ```bash
  curl -f http://localhost:${API_PORT_HOST}/health
  ```

- Airflow health:
  ```bash
  curl -f http://localhost:${AIRFLOW_PORT_HOST}/health
  ```

- MLflow root:
  ```bash
  curl -f http://localhost:${MLFLOW_PORT_HOST}/
  ```

### 4.3 UIs

- Airflow UI: `http://localhost:8080`
- MLflow UI: `http://localhost:5000`
- API docs (Swagger): `http://localhost:8000/docs`

Airflow credentials (from `.env.example`):
- user: `AIRFLOW_ADMIN_USER`
- password: `AIRFLOW_ADMIN_PASSWORD`

---

## 5) First end-to-end run (recommended path)

The project is designed to be executed through Airflow.

### 5.1 Trigger the initial setup DAG

In the Airflow UI, trigger:

- `01_news_topics_ner_initial_setup`

This DAG is intended to:
1. load initial `train` and `test` news into Postgres,
2. train the topics detector and register/log the run,
3. run full inference (topics + entities) for `train,test`.

### 5.2 Optional: start ingestion/inference simulation

The repository includes scheduled DAGs (every 10 minutes) to simulate production-like behavior:

- `03_news_topics_ner_daily_ingestion`
- `04_news_topics_ner_daily_inference`

If you want to keep things deterministic during review, you can leave them paused and only run manual DAGs.

---

## 6) Running pipelines without Airflow (optional)

You can run pipelines directly inside the API container (same Python environment as the services).

Examples:

```bash
# Initial ingestion
docker compose exec api python src/news_nlp/pipelines/01_load_initial_news_pipeline.py

# Train topics detector (requires OPENAI_API_KEY)
docker compose exec api python src/news_nlp/pipelines/02_topics_detector_train_pipeline.py

# Full inference
docker compose exec api python src/news_nlp/pipelines/05_full_inference_pipeline.py   --mode-topics-detector overwrite   --mode-ner-extractor incremental   --sources train,test
```

---

## 7) Persistence, volumes, and generated artifacts

### 7.1 Docker volumes

Two named volumes are used:

- `news_nlp_db_data`  
  Persists Postgres data across restarts (`/var/lib/postgresql/data`).

- `mlflow_artifacts`  
  Persists MLflow artifacts (`/mlflow/artifacts` in the MLflow container).

### 7.2 Repo mounts

The following bind mounts are used for reproducibility and transparency:

- API container: `./:/app`  
  Allows reviewing and running the same repository code inside the container.

- Airflow containers:
  - `./airflow_dags:/opt/airflow/dags`
  - `./:/opt/airflow/news-topics-ner`

### 7.3 Data bundle and runtime outputs

The repository includes a compressed dataset:

- `data/compressed/data.zip`

On the first ingestion run, this bundle is extracted into runtime folders (e.g., `data/raw/`, `data/processed/` depending on the pipeline logic).

Model artifacts (topics detector, spaCy model assets) are generated at runtime (typically under `models/`) and are intentionally not committed to Git.

---

## 8) Common operational commands

### View logs

```bash
docker compose logs -f api
docker compose logs -f airflow-webserver
docker compose logs -f airflow-scheduler
docker compose logs -f mlflow
docker compose logs -f db
```

### Restart a service

```bash
docker compose restart api
```

### Open a shell inside a container

```bash
docker compose exec api bash
docker compose exec airflow-webserver bash
```

### Stop everything

```bash
docker compose down
```

### Reset everything (including DB + MLflow artifacts)

**Warning:** this deletes volumes and all persisted state.

```bash
docker compose down -v
```

---

## 9) Local (non-Docker) development setup (optional)

This repo provides `.env.local.example` as a reference for running components locally.

Typical steps:
1. Create a local virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

2. Create a local `.env.local` based on `.env.local.example`:
   ```bash
   cp .env.local.example .env.local
   ```

3. Start Postgres and MLflow locally (or via Docker) and point:
   - `DB_HOST=localhost`
   - `MLFLOW_TRACKING_URI=http://localhost:5000`

This path is not the primary review path; Docker Compose is the canonical configuration for reproducibility.

---

## Related documents

- `docs/ARCHITECTURE.md`
- `docs/PIPELINES_AND_DAGS.md`
- `docs/API_REFERENCE.md`
- `docs/TROUBLESHOOTING.md`
