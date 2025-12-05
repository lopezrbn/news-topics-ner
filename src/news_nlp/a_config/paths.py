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
DF_TRAIN = DIR_DATA_RAW / "train.tsv"
DF_TEST = DIR_DATA_RAW / "test.tsv"
DF_TRAIN_CLEAN = DIR_DATA_PROCESSED / "df_train_clean.parquet"
DF_TEST_CLEAN = DIR_DATA_PROCESSED / "df_test_clean.parquet"

DF_TRAIN_CLUSTERED = DIR_DATA_PROCESSED / "df_train_clustered.parquet"
DF_ASSIGNMENTS = DIR_DATA_PROCESSED / "df_assignments.parquet"
DF_TOPICS_META = DIR_DATA_PROCESSED / "df_topics_meta.parquet"

DF_ENTITIES_RAW = DIR_DATA_PROCESSED / "df_entities_raw.parquet"
DF_ENTITIES = DIR_DATA_PROCESSED / "df_entities.parquet"
DF_NEWS_ENTITIES = DIR_DATA_PROCESSED / "df_news_entities.parquet"

# Model subdirectories
DIR_MODELS_TOPICS = DIR_MODELS / "topics"
DIR_MODELS_NER = DIR_MODELS / "ner"

# Model file paths
TFIDF_VECTORIZER = DIR_MODELS_TOPICS / "tfidf_vectorizer.joblib"
SVD_MODEL = DIR_MODELS_TOPICS / "svd_model.joblib"
KMEANS_MODEL = DIR_MODELS_TOPICS / "kmeans_model.joblib"

MODEL_NER_SPACY = DIR_MODELS_NER / "en_core_web_md"
# MODEL_NER_SPACY_SM = DIR_MODELS_NER / "en_core_web_sm"

# .env file path
ENV_FILE = DIR_BASE / ".env"