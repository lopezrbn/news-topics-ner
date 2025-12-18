# Experimentation Notebooks

This document explains the exploratory and experimentation workflow implemented in the three notebooks under `notebooks/`.

These notebooks are intentionally separated from the production pipelines:
- notebooks focus on **exploration, iteration, and configuration search**
- pipelines/DAGs implement the **reproducible, operationalized system**

Notebooks covered:
- `notebooks/01_eda.ipynb`
- `notebooks/02_topics_detector.ipynb`
- `notebooks/03_ner.ipynb`

---

## 1) Notebook 01 — EDA (`01_eda.ipynb`)

### Goal
- Understand the raw dataset structure and basic quality issues.
- Produce “clean” processed datasets that can support downstream experiments.

### Main steps

#### 1. Extract and load data
- Loads raw TSV splits from:
  - `data/raw/train.tsv` (`paths.DF_TRAIN_RAW`, defined as `DIR_DATA_RAW / "train.tsv"`)
  - `data/raw/test.tsv` (`paths.DF_TEST_RAW`, defined as `DIR_DATA_RAW / "test.tsv"`)

#### 2. Basic EDA
- Checks for missing values (NaNs) and basic field sanity.
- Inspects text length distribution:
  - helps detect very short/empty samples and extreme outliers.

#### 3. Basic preprocessing
- Applies the shared cleaning routine:
  - `news_nlp.preprocessing.text_cleaning.clean_text`
- The cleaned text is used later in both topic modeling and NER extraction.

#### 4. Export processed datasets
Exports Parquet artifacts to the processed directory:
- `data/processed/df_train_clean.parquet` (`paths.DF_TRAIN_CLEAN`, `DIR_DATA_PROCESSED / "df_train_clean.parquet"`)
- `data/processed/df_test_clean.parquet` (`paths.DF_TEST_CLEAN`, `DIR_DATA_PROCESSED / "df_test_clean.parquet"`)

These Parquet files are used as the input of the topic-model experimentation notebook.

### Output artifacts
- Cleaned splits:
  - `df_train_clean.parquet`
  - `df_test_clean.parquet`

---

## 2) Notebook 02 — Topic detector experimentation (`02_topics_detector.ipynb`)

### Goal
- Prototype an unsupervised topic detector.
- Compare configurations and select a stable baseline that can be operationalized.
- Produce intermediate artifacts useful for inspection (cluster assignments + topic summaries).

### Modeling approach
The notebook implements and validates the baseline approach used by the production pipeline:

1. **TF-IDF** vectorization (sparse features)
2. **Truncated SVD** (latent representation)
3. **KMeans** clustering (topic assignment)

### Data used
- Uses the cleaned Parquet exports from `01_eda.ipynb`:
  - `data/processed/df_train_clean.parquet`
  - `data/processed/df_test_clean.parquet`

### Main steps

#### 1. Vectorization with TF-IDF
- Runs TF-IDF vectorization to create high-dimensional sparse representations.
- Includes a “full vocabulary” run for introspection, and then bounded configurations for experiments.

#### 2. Dimensionality reduction with SVD
- Explores different `n_components` values and computes explained variance to select a reasonable latent dimensionality.

#### 3. Clustering with KMeans
- Runs KMeans over the SVD latent space.
- Produces topic/cluster assignments for the training set.

#### 4. Compare configurations (global scorecard)
The notebook includes helper utilities to compare configurations across:
- TF-IDF `max_features`
- SVD `n_components`
- KMeans `n_clusters`

Quality proxy used during configuration search:
- **silhouette score** (cluster separation / cohesion proxy)
- inertia (to understand cluster compactness trade-offs)

#### 5. Select final configuration
The notebook defines “best” hyperparameters (as constants) that are later operationalized:

- `TFIDF_BEST_MAX_FEATURES = 30_000`
- `SVD_BEST_N_COMPONENTS = 30`
- `KMEANS_BEST_N_CLUSTERS = 50`

These inform the production defaults (see `news_nlp.topics_detector.model.TopicModelConfig`).

#### 6. Export experiment artifacts
The notebook exports artifacts to support inspection and downstream prototyping:

- Clustered training set:
  - `data/processed/df_train_clustered.parquet` (`paths.DF_TRAIN_CLUSTERED`, `DIR_DATA_PROCESSED / "df_train_clustered.parquet"`)
- (Optional) assignments table (commented out in the notebook):
  - `data/processed/df_assignments.parquet` (`paths.DF_ASSIGNMENTS`, `DIR_DATA_PROCESSED / "df_assignments.parquet"`)
- Topic summary table:
  - `data/processed/df_topics.parquet` (`paths.DF_TOPICS`, `DIR_DATA_PROCESSED / "df_topics.parquet"`)

### How this notebook informs the production system
- Hyperparameter selection is carried into the production training pipeline:
  - TF-IDF max features
  - SVD components
  - KMeans cluster count
- The production pipeline reproduces the same transformations and persists:
  - model artifacts under `models/` at runtime,
  - topic dictionaries and assignments into Postgres,
  - and run metadata into MLflow.

---

## 3) Notebook 03 — NER experimentation (`03_ner.ipynb`)

### Goal
- Validate the NER extraction approach on the dataset.
- Decide the persistence layout (entities dictionary + mentions).
- Produce processed artifacts that mirror the database schema.

### NER approach
- Uses a pretrained spaCy pipeline:
  - model: `en_core_web_md`

Main actions:
- load the model,
- run entity extraction over the dataset,
- normalize entity representation,
- split results into:
  - entity dictionary
  - entity-to-news mentions.

### Main steps

#### 1. Load data
- Loads news rows from the local TSV split (the notebook uses the dataset available under `data/raw/`).

#### 2. Download and load model
- Loads `en_core_web_md` via `spacy.load(...)`.
- In the production system, the spaCy model can be loaded from runtime storage or downloaded if missing.

#### 3. Extract entities
- Runs entity extraction and collects entity mentions.

#### 4. Build entity dictionary
- Builds a unique entities table aligned with the DB table `entities`.

#### 5. Build entity-to-news mention table
- Builds a mentions table aligned with the DB table `entities_per_news`.

#### 6. Export processed artifacts
Exports Parquet files:

- Raw entities table:
  - `data/processed/df_entities_raw.parquet` (`paths.DF_ENTITIES_RAW`, `DIR_DATA_PROCESSED / "df_entities_raw.parquet"`)
- News-entity mentions table:
  - `data/processed/df_news_entities.parquet` (`paths.DF_NEWS_ENTITIES`, `DIR_DATA_PROCESSED / "df_news_entities.parquet"`)

### How this notebook informs the production system
- Confirms that spaCy provides useful entity coverage on the dataset without requiring labeled spans.
- Validates a DB-friendly persistence layout:
  - dictionary table (`entities`)
  - junction/mentions table (`entities_per_news`)
- The production pipeline reimplements the same persistence pattern directly into Postgres.

---

## 4) Relationship to pipelines and Airflow

Once notebook experimentation stabilizes choices, production code implements them in:
- training and inference pipelines (`src/news_nlp/pipelines/`)
- Airflow DAG orchestration (`airflow_dags/`)

Rule of thumb used in this repo:
- notebooks answer “**what should we do and why**”
- pipelines answer “**how do we do it reliably and repeatedly**”

---

## 5) Re-running notebooks

To run notebooks locally, you can:
- use the Docker environment (recommended for dependency parity), or
- use a local Python environment (see `docs/SETUP_AND_DEPLOYMENT.md`).

The notebooks rely on the same path configuration defined in:
- `src/news_nlp/config/paths.py`

---

## Related documents

- `docs/MODELING_AND_EVALUATION.md`
- `docs/PIPELINES_AND_DAGS.md`
- `docs/SETUP_AND_DEPLOYMENT.md`
