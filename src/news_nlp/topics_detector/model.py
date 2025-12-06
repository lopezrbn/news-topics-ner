from pathlib import Path
import sys
BASE_DIR = str(Path(__file__).resolve().parents[1])
if BASE_DIR not in sys.path:
    print(f"Adding {BASE_DIR} to sys.path")
    sys.path.insert(0, BASE_DIR)

from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from db.connection import get_engine
from config import paths


@dataclass
class TopicModelConfig:
    """
    Configuration parameters for the topics detector model.
    """
    # General
    model_name: str = "tfidf_svd_kmeans"
    top_terms_per_topic: int = 10
    random_state: int = 31415
    # TF-IDF
    tfidf_max_features: int = 30000
    tfidf_max_df: float = 0.7
    tfidf_min_df: int = 5
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_stop_words: str = "english"
    # Truncated SVD
    svd_n_components: int = 100
    # K-Means
    kmeans_n_clusters: int = 30
    kmeans_n_init: int | Literal['auto', 'warn'] = "auto"



@dataclass
class TopicModelArtifacts:
    """
    Trained artifacts and intermediate results produced by the topics detector.
    """
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    kmeans: KMeans
    X_tfidf: np.ndarray
    X_reduced: np.ndarray
    cluster_labels: np.ndarray
    silhouette: float


def load_training_news(engine: Engine | None = None) -> pd.DataFrame:
    """
    Load training news from the database.

    Returns a dataframe with columns:
      - id_news
      - text
    """
    if engine is None:
        engine = get_engine()

    query = """
        SELECT id_news, text
        FROM news
        WHERE source = 'train'
          AND text IS NOT NULL
    """

    df = pd.read_sql(query, con=engine)
    if df.empty:
        raise ValueError("No training news found in the database (source='train').")

    return df


def train_topic_model(
    texts: List[str],
    config: TopicModelConfig,
) -> TopicModelArtifacts:
    """
    Train TF-IDF + TruncatedSVD + KMeans on the given texts.
    """
    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=config.tfidf_max_features,
        max_df=config.tfidf_max_df,
        min_df=config.tfidf_min_df,
        ngram_range=config.tfidf_ngram_range,
        stop_words=config.tfidf_stop_words,
        
    )
    X_tfidf = vectorizer.fit_transform(texts)

    # SVD (dimensionality reduction)
    svd = TruncatedSVD(
        n_components=config.svd_n_components,
        random_state=config.random_state,
    )
    X_reduced = svd.fit_transform(X_tfidf)

    # K-Means
    kmeans = KMeans(
        n_clusters=config.kmeans_n_clusters,
        n_init=config.kmeans_n_init,
        random_state=config.random_state,
    )
    cluster_labels = kmeans.fit_predict(X_reduced)      # 0 to n_clusters-1

    # Silhouette score (simple quality metric)
    silhouette = silhouette_score(X_reduced, cluster_labels)

    return TopicModelArtifacts(
        vectorizer=vectorizer,
        svd=svd,
        kmeans=kmeans,
        X_tfidf=X_tfidf,
        X_reduced=X_reduced,
        cluster_labels=cluster_labels,
        silhouette=float(silhouette),
    )


def compute_top_terms_per_topic(
    config: TopicModelConfig,
    artifacts: TopicModelArtifacts,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Compute top terms most representative for each topic based on the average
    TF-IDF vector of the documents assigned to that topic.

    Returns a dict:
      id_topic -> list of (term, weight) sorted by descending weight.
    """
    X_tfidf = artifacts.X_tfidf
    cluster_labels = artifacts.cluster_labels
    vectorizer = artifacts.vectorizer

    feature_names = np.array(vectorizer.get_feature_names_out())
    n_clusters = config.kmeans_n_clusters
    top_n = config.top_terms_per_topic

    top_terms: Dict[int, List[Tuple[str, float]]] = {}

    for id_topic in range(n_clusters):
        mask = cluster_labels == id_topic
        if not np.any(mask):               # no documents assigned to this topic
            top_terms[id_topic] = []
            continue

        X_topic = X_tfidf[mask]
        mean_tfidf = X_topic.mean(axis=0).A1  # convert to 1D array

        if (mean_tfidf > 0).sum() == 0:        # no terms with positive weight for this topic
            top_terms[id_topic] = []
            continue

        top_indices = np.argsort(mean_tfidf)[::-1][:top_n]  # Sort the indices descending, based on their values, and take the top-n.
        terms = feature_names[top_indices]      # Get the terms corresponding to the top indices.
        weights = mean_tfidf[top_indices]       # Get the weights corresponding to the top indices.

        top_terms[id_topic] = list(zip(terms, weights.astype(float)))

    return top_terms


def save_topic_model_artifacts(
    id_run: int,
    artifacts: TopicModelArtifacts,
) -> None:
    """
    Save model artifacts (vectorizer, SVD, KMeans) under a directory for this run.
    """
    run_dir = paths.DIR_MODELS_TOPICS / f"run_{id_run}"
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts.vectorizer, run_dir / "tfidf_vectorizer.joblib")
    joblib.dump(artifacts.svd, run_dir / "svd.joblib")
    joblib.dump(artifacts.kmeans, run_dir / "kmeans.joblib")

    print(f"Saved topic model artifacts to {run_dir}")
