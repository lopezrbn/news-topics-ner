from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.pipeline import Pipeline

from news_nlp.preprocessing.text_cleaning import clean_text


def predict_topics_for_texts(
    texts: List[str],
    pipeline: Pipeline,
    apply_cleaning: bool = False,
    text_cleaner: Callable[[str], str] = clean_text,
) -> np.ndarray:
    """
    Predict topic ids for a list of texts using a fitted sklearn Pipeline.

    Parameters
    ----------
    texts : list of str
        Raw or already-cleaned texts.
    pipeline : Pipeline
        Fitted sklearn Pipeline (TF-IDF + SVD + KMeans).
    apply_cleaning : bool, default False
        If True, apply `text_cleaner` to each text before prediction.
        If False, texts are assumed to be already cleaned.
    text_cleaner : callable, default clean_text
        Function used to clean each text when apply_cleaning=True.

    Returns
    -------
    labels : np.ndarray
        Array of predicted topic ids (cluster labels).
    """
    if apply_cleaning:
        processed_texts = [text_cleaner(t) for t in texts]
    else:
        processed_texts = texts

    labels = pipeline.predict(processed_texts)
    return np.asarray(labels)


def predict_topic_for_text(
    text: str,
    pipeline: Pipeline,
    apply_cleaning: bool = True,
    text_cleaner: Callable[[str], str] = clean_text,
) -> int:
    """
    Predict the topic for a single text.

    Parameters
    ----------
    text : str
        Raw or already-cleaned text.
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    apply_cleaning : bool, default True
        If True, apply `text_cleaner` before prediction.
    text_cleaner : callable, default clean_text
        Cleaning function.

    Returns
    -------
    id_topic : int
    """
    if apply_cleaning:
        processed = text_cleaner(text)
    else:
        processed = text

    id_topic = int(pipeline.predict([processed])[0])

    return id_topic
