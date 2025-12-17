from pathlib import Path

# Root directory of the project
DIR_BASE = Path(__file__).parents[3].resolve()

# Subdirectories
DIR_CONFIG = DIR_BASE / "config"
DIR_DATA = DIR_BASE / "data"
DIR_DB = DIR_BASE / "db"
DIR_MODELS = DIR_BASE / "models"
DIR_NOTEBOOKS = DIR_BASE / "notebooks"
DIR_SRC = DIR_BASE / "src"

# Config file paths
PROMPTS_FILE = DIR_CONFIG / "prompts.yaml"

# DB file paths
SCHEMA_SQL_FILE = DIR_DB / "schema.sql"

# Data subdirectories
DIR_DATA_COMPRESSED = DIR_DATA / "compressed"
DIR_DATA_RAW = DIR_DATA / "raw"
DIR_DATA_PROCESSED = DIR_DATA / "processed"

# Data file paths
DATA_COMPRESSED = DIR_DATA_COMPRESSED / "data.zip"
DF_TRAIN_RAW = DIR_DATA_RAW / "train.tsv"
DF_TEST_RAW = DIR_DATA_RAW / "test.tsv"
DF_TRAIN_CLEAN = DIR_DATA_PROCESSED / "df_train_clean.parquet"
DF_TEST_CLEAN = DIR_DATA_PROCESSED / "df_test_clean.parquet"

DF_TRAIN_CLUSTERED = DIR_DATA_PROCESSED / "df_train_clustered.parquet"
DF_ASSIGNMENTS = DIR_DATA_PROCESSED / "df_assignments.parquet"
DF_TOPICS = DIR_DATA_PROCESSED / "df_topics.parquet"

DF_ENTITIES_RAW = DIR_DATA_PROCESSED / "df_entities_raw.parquet"
DF_ENTITIES = DIR_DATA_PROCESSED / "df_entities.parquet"
DF_NEWS_ENTITIES = DIR_DATA_PROCESSED / "df_news_entities.parquet"

# Model subdirectories
DIR_MODELS_TOPICS = DIR_MODELS / "topics_detector"
DIR_MODELS_NER = DIR_MODELS / "ner_extractor"

# Model file paths
TFIDF_VECTORIZER = DIR_MODELS_TOPICS / "tfidf_vectorizer.joblib"
SVD_MODEL = DIR_MODELS_TOPICS / "svd_model.joblib"
KMEANS_MODEL = DIR_MODELS_TOPICS / "kmeans_model.joblib"

MODEL_NER_SPACY = DIR_MODELS_NER / "en_core_web_md"
# MODEL_NER_SPACY_SM = DIR_MODELS_NER / "en_core_web_sm"

# Notebooks subdirectories
DIR_NOTEBOOKS_DATA = DIR_NOTEBOOKS / "data"
DIR_NOTEBOOKS_MODELS = DIR_NOTEBOOKS / "models"

# Notebooks data file paths
X_TFIDF_JLIB = DIR_NOTEBOOKS_DATA / "X_tfidf.joblib"
DF_VARIANCE = DIR_NOTEBOOKS_DATA / "df_variance.csv"
X_REDUCED_DICT_JLIB = DIR_NOTEBOOKS_DATA / "X_reduced_dict.joblib"
X_REDUCED_FINAL_JLIB = DIR_NOTEBOOKS_DATA / "X_reduced_final.joblib"
DF_SILHOUETTE_DICT_JLIB = DIR_NOTEBOOKS_DATA / "df_silhouette_dict.joblib"
DF_ALL_TOPICS_AND_CONFIGS = DIR_NOTEBOOKS_DATA / "df_all_topics_and_configs.parquet"
DF_GRID_SEARCH = DIR_NOTEBOOKS_DATA / "df_grid_search.csv"
TOPIC_NAMES_FILE = DIR_NOTEBOOKS_DATA / "topic_names.json"

# Notebooks models file paths
MODEL_NOTEBOOKS_TFIDF = DIR_NOTEBOOKS_MODELS / "vectorizer_tfidf.joblib"
MODEL_NOTEBOOKS_SVD = DIR_NOTEBOOKS_MODELS / "reductor_svd.joblib"
MODEL_NOTEBOOKS_KMEANS = DIR_NOTEBOOKS_MODELS / "clusterer_kmeans.joblib"
MODEL_NOTEBOOKS_PIPELINE = DIR_NOTEBOOKS_MODELS / "model_pipeline.joblib"

# .env file path
ENV_FILE = DIR_BASE / ".env"
# ENV_FILE = DIR_BASE / ".env.local"