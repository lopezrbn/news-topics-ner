# Pipelines and DAGs

This document explains the **end-to-end execution flow** of the project:
- what each pipeline does,
- how pipelines are parameterized (CLI),
- and how Airflow DAGs orchestrate those pipelines over time.

The repository supports three logical data sources:
- `train` and `test` (batch dataset splits)
- `prod` (a simulated “production” stream ingested incrementally)

---

## 1) Execution layers

### 1.1 Pipelines (entry points)

Pipeline entry points live in:

- `src/news_nlp/pipelines/*.py`

They are thin scripts that:
1. load environment variables (from `.env`),
2. parse CLI arguments when needed,
3. call a job function under `src/news_nlp/pipelines/jobs/`.

### 1.2 Jobs (business logic)

Jobs live in:

- `src/news_nlp/pipelines/jobs/*.py`

They implement the actual logic:
- load rows from Postgres,
- run the topics detector and/or NER model,
- write results back to Postgres,
- and (for training) log to MLflow and save artifacts to disk.

### 1.3 Airflow DAGs (orchestration)

Airflow DAGs live in:

- `airflow_dags/*.py`

They orchestrate pipelines either:
- via **BashOperator** (execute pipeline scripts), or
- via **PythonOperator** (directly call a Python function for ingestion simulation).

---

## 2) Pipeline catalog

### 2.1 `01_load_initial_news_pipeline.py`

**Purpose**
- Load the initial `train` and `test` splits into Postgres (`news` table).

**Where it reads from**
- `data/raw/train.tsv`
- `data/raw/test.tsv`

If those files are missing, ingestion automatically extracts:
- `data/compressed/data.zip` into `data/raw/` (via `news_nlp.ingestion.db_io._extract_data_from_zip`).

**Core steps**
1. Ensure raw TSVs exist (extract from zip if needed).
2. Read TSV.
3. Build a `text` field from `title` and `content`.
4. Clean the text (`news_nlp.preprocessing.text_cleaning.clean_text`).
5. Insert into Postgres:
   - `source='train'` for train split
   - `source='test'` for test split

**Outputs**
- Rows in `news` table for `train` and `test`.

---

### 2.2 `02_topics_detector_train_pipeline.py`

**Purpose**
- Train the **topic detector** model and register the run.

**Model approach**
- TF-IDF vectorizer
- Truncated SVD
- KMeans clustering

Config defaults are defined in `news_nlp.topics_detector.model.TopicModelConfig`, e.g.:
- `tfidf_max_features=30000`
- `svd_n_components=30`
- `kmeans_n_clusters=50`

**Core steps**
1. Load env vars and connect to Postgres.
2. Initialize MLflow (`news_nlp.config.mlflow_config.init_mlflow`).
3. Load training texts from DB (`source='train'`).
4. Start an MLflow run.
5. Train the pipeline and compute:
   - clustering labels
   - silhouette score (and other diagnostics)
   - top terms per topic
6. (Optional) Generate human-readable topic names via LLM prompts (`config/prompts.yaml`) using `OPENAI_API_KEY`.
7. Persist run metadata:
   - insert a row in `topics_model_training_runs`
   - insert/update dictionaries:
     - `topics`
     - `terms_per_topic`
8. Save model artifacts to disk under `models/topics_detector/...` (generated at runtime; not committed).

**Outputs**
- A new (or updated) topics training run in `topics_model_training_runs`
- Topic dictionary tables (`topics`, `terms_per_topic`)
- Model artifacts under `models/`
- MLflow run with parameters/metrics/artifacts

---

### 2.3 `03_topics_detector_inference_pipeline.py`

**Purpose**
- Run topics inference for news items and write topic assignments.

**CLI**
This pipeline supports:
- selecting sources (`--sources`)
- selecting the run (`--id-run`) or using the active run
- setting inference mode (`--mode-topics-detector`)

**Core steps**
1. Load env vars.
2. Resolve `id_run`:
   - if `--id-run` is omitted, uses the active run from DB (`is_active=true`).
3. Load topic detector pipeline artifacts for `id_run`.
4. Load candidate news rows from DB (see “Modes” below).
5. Predict cluster/topic labels.
6. Write results to Postgres (`topics_per_news`), referencing `id_run`.

**Outputs**
- New rows in `topics_per_news` (or refreshed rows if overwrite mode is used).

---

### 2.4 `04_ner_extractor_inference_pipeline.py`

**Purpose**
- Run NER extraction for news items and write entities + mentions.

**NER approach**
- spaCy model loaded via `news_nlp.ner_extractor.model.load_spacy_model`
- Entities are extracted and normalized; unique entities are maintained in `entities`,
  and mentions are stored in `entities_per_news`.

**CLI**
- `--sources` (optional)
- `--mode-ner-extractor` (currently only `incremental` is supported)

**Core steps**
1. Load env vars.
2. Load spaCy model (download if missing; runtime generated).
3. Load candidate news rows from DB (incremental mode only).
4. Extract entity mentions.
5. Build unique entities and insert into `entities`.
6. Fetch mapping `(entity_text_norm, entity_type) -> id_entity`.
7. Save mentions into `entities_per_news`.

**Outputs**
- New rows in `entities` (unique dictionary)
- New rows in `entities_per_news` (mentions per news)

---

### 2.5 `05_full_inference_pipeline.py`

**Purpose**
- Convenience pipeline that runs:
  1) topics inference
  2) NER inference

It calls `run_full_inference_job`, which internally runs:
- `run_topics_detector_inference_job`
- `run_ner_extractor_inference_job`

This is the pipeline used by most Airflow DAGs.

---

## 3) CLI reference (inference pipelines)

Inference CLI parsing is implemented in:

- `src/news_nlp/pipelines/jobs/cli_utils.py`

### 3.1 `--sources`

```text
--sources train
--sources train,test
--sources prod
--sources all
```

Semantics:
- `all` means “do not filter by source” (process any source).
- Otherwise, a comma-separated list is interpreted as an explicit source filter.

Valid sources in this repository:
- `train`, `test`, `prod`

### 3.2 `--id-run` (topics inference)

```text
--id-run 7
```

If omitted:
- `03_topics_detector_inference_pipeline.py` and `05_full_inference_pipeline.py` will use the **active** run id (`is_active=true`) from `topics_model_training_runs`.

### 3.3 `--mode-topics-detector`

```text
--mode-topics-detector incremental
--mode-topics-detector overwrite
```

- `incremental`: process only news that do **not** yet have a `topics_per_news` row for the target run.
- `overwrite`: delete existing assignments for `id_run` (and selected sources) and recompute.

### 3.4 `--mode-ner-extractor`

```text
--mode-ner-extractor incremental
```

Only `incremental` is supported:
- process only news that do **not** yet have rows in `entities_per_news`.

---

## 4) Airflow DAGs

Airflow DAGs are located in `airflow_dags/`.

### Shared runtime configuration

Some DAGs support environment-variable overrides:

- `NEWS_NLP_DIR_BASE`  
  Base directory where the repository is mounted inside the Airflow container.

- `NEWS_NLP_VENV_PYTHON`  
  Python interpreter to use when executing pipeline scripts via BashOperator.

In Docker Compose, these are set for Airflow containers so that:
- `NEWS_NLP_DIR_BASE=/opt/airflow/news-topics-ner`
- `NEWS_NLP_VENV_PYTHON=python`
- `PYTHONPATH=/opt/airflow/news-topics-ner/src`

---

### 4.1 DAG 01 — Initial setup

File:
- `airflow_dags/01_news_topics_ner_initial_setup_dag.py`

DAG id:
- `01_news_topics_ner_initial_setup`

Schedule:
- `None` (manual trigger)

Tasks (BashOperator):
1. `load_initial_news`  
   Runs:
   ```bash
   python src/news_nlp/pipelines/01_load_initial_news_pipeline.py
   ```

2. `train_topics_detector`  
   Runs:
   ```bash
   python src/news_nlp/pipelines/02_topics_detector_train_pipeline.py
   ```

3. `full_inference_initial`  
   Runs:
   ```bash
   python src/news_nlp/pipelines/05_full_inference_pipeline.py      --mode-topics-detector overwrite      --mode-ner-extractor incremental      --sources train,test
   ```

Dependencies:
- `load_initial_news >> train_topics_detector >> full_inference_initial`

---

### 4.2 DAG 02 — Retrain

File:
- `airflow_dags/02_news_topics_retrain_dag.py`

DAG id:
- `02_news_topics_ner_retrain`

Schedule:
- `None` (manual trigger)

Tasks (BashOperator):
1. `train_topics_detector`  
   Runs:
   ```bash
   python src/news_nlp/pipelines/02_topics_detector_train_pipeline.py
   ```

2. `full_inference_after_retrain`  
   Runs:
   ```bash
   python src/news_nlp/pipelines/05_full_inference_pipeline.py      --mode-topics-detector overwrite      --mode-ner-extractor incremental      --sources all
   ```

Dependency:
- `train_topics_detector >> full_inference_after_retrain`

---

### 4.3 DAG 03 — Hourly ingestion simulation

File:
- `airflow_dags/03_news_topics_ner_daily_ingestion_dag.py`

DAG id:
- `03_news_topics_ner_daily_ingestion`

Schedule:
- `0 * * * *` (hourly)

Start date:
- `2025-12-01T00:00:00+00:00` (UTC)

Task (PythonOperator):
- `ingest_simulated_news_prod`

It calls `ingest_task(...)`, which then calls:

- `news_nlp.ingestion.simulated_ingestion.load_fraction_prod_into_news_table(...)`

Key parameters (from DAG code):
- `tsv_path`: `paths.DF_TEST_RAW` (uses `test.tsv` as the simulation source)
- `fraction_per_run`: `0.01` (1% of TSV per run)
- `period_seconds`: `3600` (hourly alignment)
- `start_logical_dt_iso`: `"2025-12-01T00:00:00+00:00"`

How the simulation works (high level):
1. The DAG’s `logical_date` determines a **run index** relative to `START_LOGICAL_DT_ISO`.
2. Each run selects a deterministic slice of rows `[start_row, end_row)` from the TSV.
3. The slice is written to a temporary TSV and ingested into Postgres with `source='prod'`.

Operational note:
- The task sets `retries=0` to reduce the chance of duplicate inserts on retry.

---

### 4.4 DAG 04 — Daily inference on prod

File:
- `airflow_dags/04_news_topics_ner_daily_inference_dag.py`

DAG id:
- `04_news_topics_ner_daily_inference`

Schedule:
- `0 7 * * *` (daily at 07:00)

Task (BashOperator):
- `full_inference_incremental`

Runs:
```bash
python src/news_nlp/pipelines/05_full_inference_pipeline.py   --mode-topics-detector incremental   --mode-ner-extractor incremental   --sources prod
```

---

## 5) Data touchpoints (DB tables)

The pipelines interact with the business schema (`news_nlp` DB) at these main tables:

- `news`
  - ingestion writes `train/test/prod` rows here

- `topics_model_training_runs`
  - training registers runs (`id_run`), MLflow id, and active run status

- `topics`, `terms_per_topic`
  - topic dictionary tables created/updated during training

- `topics_per_news`
  - topics assignments (cluster label per news per run)

- `entities`, `entities_per_news`
  - entity dictionary + mentions per news

---

## 6) Recommended execution flows

### First-time run (clean clone)
1. Bring up Docker Compose stack.
2. Trigger `01_news_topics_ner_initial_setup`.
3. Inspect:
   - Airflow task logs
   - MLflow experiment UI
   - DB tables for expected rows

### After model changes / experimentation
1. Trigger `02_news_topics_ner_retrain`.
2. Optionally rerun full inference in overwrite mode for a clean state.

### Production-like simulation (optional)
1. Unpause (or manually run) `03_news_topics_ner_daily_ingestion`.
2. Run `04_news_topics_ner_daily_inference` to process newly ingested `prod` data.

---

## Related documents

- `docs/ARCHITECTURE.md`
- `docs/SETUP_AND_DEPLOYMENT.md`
- `docs/MODELING_AND_EVALUATION.md`
- `docs/API_REFERENCE.md`
- `docs/TROUBLESHOOTING.md`
