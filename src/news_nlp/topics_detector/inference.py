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
    topic_id_to_name: Optional[Dict[int, str]] = None,
    apply_cleaning: bool = True,
    text_cleaner: Callable[[str], str] = clean_text,
) -> Dict[str, Optional[object]]:
    """
    Predict the topic for a single text.

    Parameters
    ----------
    text : str
        Raw or already-cleaned text.
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    topic_id_to_name : dict[int, str], optional
        Optional mapping from topic_id to human-readable topic_name.
        If provided, the returned dict will include topic_name.
    apply_cleaning : bool, default True
        If True, apply `text_cleaner` before prediction.
    text_cleaner : callable, default clean_text
        Cleaning function.

    Returns
    -------
    result : dict
        Keys:
          - "topic_id": int
          - "topic_name": Optional[str]
    """
    if apply_cleaning:
        processed = text_cleaner(text)
    else:
        processed = text

    topic_id = int(pipeline.predict([processed])[0])

    topic_name = None
    if topic_id_to_name is not None:
        topic_name = topic_id_to_name.get(topic_id)

    return {
        "topic_id": topic_id,
        "topic_name": topic_name,
    }
