from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from news_nlp.config import paths


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
    pipeline: Pipeline
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    kmeans: KMeans
    X_tfidf: np.ndarray
    X_reduced: np.ndarray
    cluster_labels: np.ndarray
    silhouette: float


def load_topic_pipeline(id_run: int) -> Pipeline:
    """
    Load the fitted sklearn Pipeline (TF-IDF + SVD + KMeans) for the given run.
    """
    run_dir = paths.DIR_MODELS_TOPICS / f"run_{str(id_run).zfill(2)}"
    pipeline_path = run_dir / "topics_detector_pipeline.joblib"

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline file not found at {pipeline_path}. "
            "Make sure you have saved it in the training step."
        )

    pipeline: Pipeline = joblib.load(pipeline_path)
    return pipeline


def train_topic_model(
    texts: List[str],
    config: TopicModelConfig,
) -> TopicModelArtifacts:
    """
    Train the topics detector model.

    This function:
      - builds a TF-IDF + SVD + KMeans pipeline,
      - fits the pipeline on the given texts,
      - computes intermediate matrices and the silhouette score,
      - returns all artifacts in a TopicModelArtifacts instance.

    Parameters
    ----------
    texts : list of str
        Clean texts (already preprocessed) to cluster.
    config : TopicModelConfig
        Hyperparameters for the model.

    Returns
    -------
    artifacts : TopicModelArtifacts
    """
    # 1) Build components
    vectorizer = TfidfVectorizer(
        max_features=config.tfidf_max_features,
        max_df=config.tfidf_max_df,
        min_df=config.tfidf_min_df,
        ngram_range=config.tfidf_ngram_range,
        stop_words=config.tfidf_stop_words,
    )

    svd = TruncatedSVD(
        n_components=config.svd_n_components,
        random_state=config.random_state,
    )

    kmeans = KMeans(
        n_clusters=config.kmeans_n_clusters,
        n_init=config.kmeans_n_init,
        random_state=config.random_state,
    )

    # 2) Build sklearn Pipeline (without text cleaning: texts must be already clean)
    pipeline = Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("svd", svd),
            ("kmeans", kmeans),
        ]
    )

    # 3) Fit pipeline on the input texts
    pipeline.fit(texts)

    # 4) Retrieve fitted components from the pipeline
    fitted_vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    fitted_svd: TruncatedSVD = pipeline.named_steps["svd"]
    fitted_kmeans: KMeans = pipeline.named_steps["kmeans"]

    # 5) Compute intermediate matrices for inspection / metrics
    X_tfidf = fitted_vectorizer.transform(texts)
    X_reduced = fitted_svd.transform(X_tfidf)
    cluster_labels = fitted_kmeans.labels_

    if len(np.unique(cluster_labels)) > 1:
        silhouette = float(silhouette_score(X_reduced, cluster_labels))
    else:
        silhouette = -1.0

    artifacts = TopicModelArtifacts(
        pipeline=pipeline,
        vectorizer=fitted_vectorizer,
        svd=fitted_svd,
        kmeans=fitted_kmeans,
        X_tfidf=X_tfidf,
        X_reduced=X_reduced,
        cluster_labels=cluster_labels,
        silhouette=silhouette,
    )

    return artifacts


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
) -> Dict[str, paths.Path]:
    """
    Save model artifacts (pipeline, vectorizer, SVD, KMeans) under a directory for this run.
    """
    
    # Create run directory
    dir_run = paths.DIR_MODELS_TOPICS / f"run_{str(id_run).zfill(2)}"
    dir_run.mkdir(parents=True, exist_ok=True)

    # 
    artifact_paths = {
        "tfidf_vectorizer": dir_run / "tfidf_vectorizer.joblib",
        "svd": dir_run / "svd.joblib",
        "kmeans": dir_run / "kmeans.joblib",
        "topics_detector_pipeline": dir_run / "topics_detector_pipeline.joblib",
    }

    # Save individual components
    joblib.dump(artifacts.vectorizer, artifact_paths["tfidf_vectorizer"])
    joblib.dump(artifacts.svd, artifact_paths["svd"])
    joblib.dump(artifacts.kmeans, artifact_paths["kmeans"])

    # Save the full sklearn Pipeline
    joblib.dump(artifacts.pipeline, artifact_paths["topics_detector_pipeline"])

    print(f"Saved topic model artifacts to {dir_run}")

    return artifact_paths
