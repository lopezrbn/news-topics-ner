# Decisions (ADR-style)

This document records key technical decisions made in **news-topics-ner**.

The format is lightweight “ADR-style”:
- **Context**: what problem we were solving
- **Decision**: what we chose
- **Alternatives**: what else we could have done
- **Consequences**: trade-offs and implications

---

## ADR-001 — Use Docker Compose as the canonical execution environment

### Context
The project is evaluated as a technical assignment, so reviewers should be able to run it from a clean GitHub clone with minimal setup variance.

### Decision
Use `docker compose` as the canonical environment for:
- Postgres
- MLflow
- FastAPI
- Airflow

### Alternatives
- Local Python environment + local Postgres
- Conda-based environment
- Kubernetes / Helm

### Consequences
- High reproducibility for reviewers
- Single-host and local-only by design (not production-grade deployment)
- Requires Docker availability

---

## ADR-002 — Use Airflow DAGs for orchestration instead of ad-hoc scripts

### Context
The system includes multiple steps (ingestion → training → inference → scheduled updates). A reviewer should see a clear operational story and the ability to run workflows deterministically.

### Decision
Use Apache Airflow DAGs to orchestrate:
- initial setup (manual)
- retraining (manual)
- simulated ingestion (scheduled)
- incremental inference (scheduled)

### Alternatives
- Cron jobs
- A Python “runner” script with CLI
- Prefect / Dagster

### Consequences
- Clear, reviewable, observable workflows (UI + logs)
- More moving parts than a simple script (Airflow DB, scheduler/webserver)
- DAG correctness becomes part of the deliverable (but also a positive signal)

---

## ADR-003 — Use MLflow for experiment tracking and artifact management

### Context
Topic model training should be inspectable and comparable across runs. Reviewers often look for run traceability and reproducibility.

### Decision
Use MLflow as the tracking server for:
- parameters
- metrics (e.g., silhouette score)
- artifacts (trained vectorizer/SVD/KMeans pipeline, topic summaries)

### Alternatives
- Logging to JSON/CSV files
- Weights & Biases
- Neptune
- No tracking (single run)

### Consequences
- Strong visibility into training runs
- Additional service + configuration (tracking URI, backend store, artifacts volume)
- Keeps model run history beyond a single execution

---

## ADR-004 — Store business data and inference outputs in Postgres

### Context
The system needs a canonical store for:
- ingested news
- active training run metadata
- topic dictionaries and topic assignments
- entity dictionaries and entity mentions

### Decision
Use Postgres as the business database (`news_nlp`), and store:
- `news`
- `topics_model_training_runs`
- `topics`, `terms_per_topic`, `topics_per_news`
- `entities`, `entities_per_news`

### Alternatives
- Flat files (CSV/Parquet)
- SQLite
- Document store (MongoDB)
- A feature store

### Consequences
- Enables consistent joins, incremental processing, and run linkage
- Requires DB schema maintenance
- Great fit for local technical evaluation; production scale would require tuning and indexing

---

## ADR-005 — Use a single Postgres instance with three databases

### Context
Airflow and MLflow both require a relational backend. The project also needs a business DB.

### Decision
Use one Postgres container hosting three databases:
- `news_nlp` (business)
- `airflow_db` (Airflow metadata)
- `mlflow_db` (MLflow backend store)

### Alternatives
- Separate Postgres containers (one per service)
- External managed DB
- SQLite backends (not recommended for Airflow/MLflow in most cases)

### Consequences
- Simple local setup and one DB service to manage
- Logical separation via databases, reduced risk of table collision
- In production, separate instances may be preferred for isolation/scaling

---

## ADR-006 — Topic modeling approach: TF-IDF + SVD + KMeans (unsupervised)

### Context
The project requires topic detection without labeled topics. The model should be:
- explainable,
- deterministic enough for review,
- easy to serialize and load for inference.

### Decision
Use:
- TF-IDF vectorizer
- Truncated SVD
- KMeans clustering

### Alternatives
- LDA
- BERTopic (embeddings + clustering)
- Transformer embeddings + HDBSCAN + c-TF-IDF

### Consequences
- Strong baseline with clear artifacts and predictable inference
- Topic quality depends on preprocessing and choice of `k`
- Advanced approaches could improve semantics but increase complexity and runtime

---

## ADR-007 — Optional LLM-assisted topic naming

### Context
Unsupervised clusters are not inherently “named”. Reviewers benefit from human-readable topic labels, but the project should remain runnable without external dependencies when possible.

### Decision
Support optional topic naming via LLM:
- prompts in `config/prompts.yaml`
- code in `news_nlp.topics_detector.topics_naming`
- guarded by `OPENAI_API_KEY`

### Alternatives
- No naming (only topic ids + top terms)
- Rule-based naming from top terms
- Manual naming in notebooks

### Consequences
- Better interpretability for demos and evaluation
- Requires an API key (and therefore is optional by design)
- Even without naming, the system remains interpretable via top terms

---

## ADR-008 — NER via pretrained spaCy model

### Context
The project requires entity extraction but does not ship labeled training data for NER.

### Decision
Use a pretrained spaCy NER model:
- loaded/downloaded at runtime
- entities persisted into dictionary + mentions tables

### Alternatives
- Train a custom NER model (requires labeled spans)
- Use a transformer NER pipeline (e.g., Hugging Face)
- Use an LLM for entity extraction

### Consequences
- Strong and simple baseline, easy to run and explain
- NER quality depends on domain mismatch between pretrained model and dataset
- Formal evaluation is not included unless labeled data is added

---

## ADR-009 — Do not commit model artifacts to Git

### Context
Model artifacts can be large and change frequently between runs. The goal is a clean repo that can be reproduced from scratch.

### Decision
Generate artifacts at runtime (typically under `models/`) and keep them out of version control.

### Alternatives
- Commit small artifacts directly
- Use Git LFS
- Upload artifacts to a model registry / object store

### Consequences
- Keeps the repo lightweight and reviewer-friendly
- Requires running training at least once before API topics inference works
- Encourages correct artifact lifecycle management (training → persistence → inference)

---

## ADR-010 — Simulated prod ingestion via deterministic slicing

### Context
A technical assignment benefits from showing:
- scheduled ingestion,
- scheduled inference,
- incremental processing,
without requiring an external news feed.

### Decision
Implement a deterministic ingestion simulator:
- slices a fixed TSV by logical time windows
- ingests a small fraction per run into `source='prod'`
- orchestrated by an Airflow scheduled DAG

### Alternatives
- Random sampling on each run
- External API ingestion
- Disable prod simulation entirely

### Consequences
- Deterministic behavior (easy to reason about during review)
- Possible duplicates if retries/backfills occur (handled operationally via DAG config)
- Demonstrates a production-like orchestration pattern within a self-contained repo
