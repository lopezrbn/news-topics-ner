# Modeling and Evaluation

This document describes:
- the data used in this repository,
- the preprocessing steps,
- the modeling approach for topic detection and NER,
- and how training/inference are evaluated and tracked.

This project is designed for a technical evaluation setting (clarity and reproducibility over production hardening).

---

## 1) Data

### 1.1 Dataset format

The repository includes a compressed dataset bundle:

- `data/compressed/data.zip`

On first ingestion, the pipelines extract it into:

- `data/raw/train.tsv`
- `data/raw/test.tsv`

Each TSV contains news items with at least:
- `title`
- `content`

The ingestion pipeline builds a single text field from these columns:

- `text = title + ". " + content`

### 1.2 Data sources in the system

The business database stores news rows in the `news` table with a `source` label:

- `train`: training split loaded from `train.tsv`
- `test`: evaluation split loaded from `test.tsv`
- `prod`: simulated streaming ingestion (incremental slices from TSV)

The `prod` stream is meant to emulate a production feed for operational testing (scheduled ingestion + scheduled inference).

---

## 2) Preprocessing

### 2.1 Text cleaning

Text is normalized through:

- `news_nlp.preprocessing.text_cleaning.clean_text`

The cleaning function targets typical noise in raw news text:
- trimming whitespace
- normalizing line breaks
- removing duplicated spaces
- (optionally) removing or normalizing non-alphanumeric characters

The result is a cleaned string suitable for:
- TF-IDF vectorization (topic model)
- spaCy tokenization (NER)

### 2.2 Train/inference consistency

The same cleaned `text` field is used across:
- topic model training,
- topics inference,
- NER extraction.

This ensures consistent tokenization and feature extraction between training and inference.

---

## 3) Topic detection model

### 3.1 Objective

The topic detection component aims to assign each news item to a discrete topic cluster and provide:
- a topic id/label (`id_topic`),
- top terms per topic (for interpretability),
- optional human-readable topic names.

### 3.2 Modeling approach

The pipeline uses a classical unsupervised approach that is reliable and easy to operationalize:

1. **TF-IDF vectorization**
   - Converts text into sparse term-weight vectors.

2. **Dimensionality reduction (Truncated SVD)**
   - Reduces sparse TF-IDF vectors into a dense latent space.
   - Enables more stable clustering and reduces noise.

3. **Clustering (KMeans)**
   - Assigns each item to one of `k` clusters.
   - Cluster id is treated as the topic id.

The configuration defaults are defined in:
- `news_nlp.topics_detector.model.TopicModelConfig`

Example defaults used in this repository:
- `tfidf_max_features=30000`
- `svd_n_components=30`
- `kmeans_n_clusters=50`

### 3.3 Artifacts produced

Training produces the following artifacts (generated at runtime, not committed to Git):

- `tfidf_vectorizer.joblib`
- `svd.joblib`
- `kmeans.joblib`
- `topics_detector_pipeline.joblib` (end-to-end pipeline)

These artifacts are saved under:

- `models/topics_detector/<run_id>/...`

### 3.4 Topic interpretability

Interpretability is supported via two mechanisms:

#### 3.4.1 Top terms per topic (deterministic)

For each cluster/topic, the pipeline derives:
- top TF-IDF terms
- and stores them in Postgres (`terms_per_topic`)

This provides a stable interpretation surface independent of any LLM.

#### 3.4.2 Topic naming (optional, LLM-assisted)

The repository supports optional topic naming via LLM:
- prompt templates in `config/prompts.yaml`
- code path in `news_nlp.topics_detector.topics_naming`

This step requires:
- `OPENAI_API_KEY`

The output is stored in:
- `topics.topic_name`

If no API key is configured, topic naming can be skipped (topic ids and terms remain available).

---

## 4) Named Entity Recognition (NER)

### 4.1 Objective

Given a news item, extract entity mentions and persist:
- the entity dictionary (unique entities) and
- entity-to-news mentions.

Entities include typical spaCy labels such as:
- `PERSON`, `ORG`, `GPE`, `DATE`, etc.

### 4.2 Modeling approach

NER is implemented using spaCy:

- the model is loaded through `news_nlp.ner_extractor.model.load_spacy_model`
- the pipeline extracts entities from the cleaned text

The runtime can:
- load a local model from `models/ner_extractor/...` if present, or
- download the configured spaCy model if missing (runtime-generated).

### 4.3 Persistence layout

NER results are stored in two tables:

- `entities`: unique entity records (normalized string + type)
- `entities_per_news`: mentions of entities per news item

This structure avoids duplicating the same entity string across many news rows.

---

## 5) Evaluation

This repository prioritizes clarity and reproducibility. Evaluation is implemented in two layers:

1) **Intrinsic diagnostics** during training (topic model quality proxies)  
2) **Operational validation** of inference outputs persisted to Postgres

### 5.1 Topic model diagnostics

The training pipeline computes (at minimum):
- **silhouette score** on the latent space (SVD outputs)

Silhouette score is a proxy for:
- cluster separation
- cluster cohesion

Additional diagnostics commonly used (and reasonable next steps) include:
- topic size distribution (cluster imbalance)
- stability across runs (agreement between runs)
- manual interpretability review (top terms and sampled articles per cluster)

### 5.2 NER quality

NER here uses a pretrained spaCy model. Formal evaluation would require labeled entity spans, which is not included in this repo.

For this technical project, quality is validated through:
- deterministic extraction behavior,
- persistence correctness (dictionary + mentions),
- and spot checks in notebooks (`notebooks/03_ner.ipynb`).

If you wanted to add formal evaluation later:
- introduce a labeled validation subset,
- compute entity-level precision/recall/F1 using spaCy’s scorer or seqeval.

### 5.3 End-to-end validation

The end-to-end “correctness” can be validated by:

- Running the initial setup DAG
- Checking that:
  - `news` contains `train/test` rows
  - `topics_model_training_runs` has a new active run
  - `topics_per_news` contains topic assignments for `train/test`
  - `entities_per_news` contains entity mentions for `train/test`

For the production simulation:
- `03_news_topics_ner_daily_ingestion` should append `prod` rows
- `04_news_topics_ner_daily_inference` should incrementally fill `topics_per_news` and `entities_per_news` for `prod`

---

## 6) Experiment tracking (MLflow)

MLflow is used to make training runs inspectable and reproducible.

### 6.1 What is logged

The topics training pipeline logs:
- parameters (TF-IDF / SVD / KMeans configuration)
- metrics (e.g., silhouette score)
- artifacts:
  - trained model objects
  - topic summaries (optional)

Run metadata is also registered into Postgres (`topics_model_training_runs`), enabling the rest of the system to:
- select an active run,
- and link inference outputs to a run id (`id_run`).

### 6.2 How to inspect runs

1. Start the Docker stack.
2. Open MLflow UI:
   - `http://localhost:5000`
3. Browse experiments and runs.
4. Compare parameters/metrics across runs.

---

## 7) Recommended next steps (optional improvements)

If you have time to extend the project, these are high-signal improvements for an ML Engineer review:

### Modeling improvements
- Use more robust topic modeling approaches:
  - BERTopic (embeddings + clustering + c-TF-IDF)
  - sentence-transformer embeddings + HDBSCAN
- Add automated topic coherence metrics (e.g., NPMI / c_v) if feasible.

### Evaluation improvements
- Add a lightweight human evaluation report:
  - sample N articles per topic
  - check interpretability and cluster consistency
- Add basic regression checks:
  - topic distribution drift across runs
  - number of empty/short texts
- For NER, add a tiny labeled set for span-level metrics.

### MLOps improvements
- Add unit tests for:
  - DB I/O functions
  - CLI parsing behavior
  - deterministic slicing in ingestion simulation
- Add CI workflow (lint + tests) if publishing publicly.

---

## Related documents

- `docs/ARCHITECTURE.md`
- `docs/PIPELINES_AND_DAGS.md`
- `docs/API_REFERENCE.md`
- `docs/TROUBLESHOOTING.md`
