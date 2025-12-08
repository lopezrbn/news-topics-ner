from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd
import spacy
from spacy.language import Language
from spacy.cli import download as spacy_download

from news_nlp.config import paths


@dataclass
class NerModelConfig:
    """
    Configuration parameters for the NER extractor.

    spacy_model_name:
        Name of the spaCy model to use (must be installed, e.g. 'en_core_web_sm').
    entity_types_to_keep:
        Optional list of entity labels to keep (e.g. ['PERSON', 'ORG', 'GPE', 'LOC']).
        If None, all entity types returned by the model are kept.
    batch_size:
        Batch size used in spaCy's nlp.pipe for faster processing.
    n_process:
        Number of processes to use in spaCy's nlp.pipe.
    require_gpu:
        If True, require a GPU and raise an error if not available.
    """

    spacy_model_name: str = "en_core_web_md"
    entity_types_to_keep: Optional[List[str]] = None
    batch_size: int = 64
    n_process: int = 4
    require_gpu: bool = False


def load_spacy_model(config: NerModelConfig) -> Language:
    """
    Load the spaCy model specified in the configuration from disk or download
    it if not available.

    Behavior:
      1) If a local copy exists under DIR_MODELS_NER / spacy_model_name, load from there.
      2) Otherwise, try to load the installed spaCy package by name.
      3) If that fails, download the model via spaCy CLI, then load it.
      4) Save the loaded model to the local directory for future runs.
      5) Optionally require GPU if config.require_gpu is True.
    """
    model_name = config.spacy_model_name
    model_local_dir = paths.DIR_MODELS_NER / model_name

    # 1) Try to load from local directory
    if model_local_dir.exists():
        nlp = spacy.load(str(model_local_dir))
        print(f"Loaded spaCy model from disk at {model_local_dir}.")
    else:
        # 2) Try to load installed spaCy model by name
        try:
            nlp = spacy.load(model_name)
            print(f"Loaded spaCy model '{model_name}' from environment.")
        except OSError:
            # 3) Download via spaCy CLI and load
            print(f"spaCy model '{model_name}' not found. Downloading...")
            spacy_download(model_name)
            nlp = spacy.load(model_name)
            print(f"Downloaded and loaded spaCy model '{model_name}'.")

        # 4) Save to local directory for future runs
        model_local_dir.parent.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(model_local_dir)
        print(f"Saved spaCy model to {model_local_dir}.")

    # 5) Optionally require GPU
    if config.require_gpu:
        spacy.require_gpu()
        print("spaCy is now configured to use GPU.")

    return nlp


def extract_entities_for_news(
    df_news: pd.DataFrame,
    nlp: Language,
    config: NerModelConfig,
) -> pd.DataFrame:
    """
    Run NER on all news and return a 'mentions' dataframe.

    Parameters
    ----------
    df_news : DataFrame
        Must contain columns:
          - id_news
          - text
    nlp : Language
        Loaded spaCy model.
    config : NerModelConfig
        NER configuration.

    Returns
    -------
    df_mentions : DataFrame
        Columns:
          - id_news
          - entity_text
          - entity_type
    """
    ids = df_news["id_news"].tolist()
    texts = df_news["text"].tolist()

    rows = []

    # Precompute allowed labels (if any)
    if config.entity_types_to_keep is not None:
        allowed_labels = set(config.entity_types_to_keep)
    else:
        allowed_labels = None

    # Extract entities in batches using nlp.pipe
    for id_news, doc in zip(
        ids,
        nlp.pipe(
            texts,
            batch_size=config.batch_size,
            n_process=config.n_process,
        ),
    ):
        for ent in doc.ents:
            # If a filter is defined, skip entities not in the allowed set
            if allowed_labels is not None and ent.label_ not in allowed_labels:
                continue

            rows.append(
                {
                    "id_news": int(id_news),
                    "entity_text": ent.text,
                    "entity_type": ent.label_,
                    # "start_char": ent.start_char,
                    # "end_char": ent.end_char,
                }
            )

    df_mentions = pd.DataFrame(rows)

    return df_mentions


def extract_entities_for_text(
    text: str,
    nlp: Language,
    config: NerModelConfig,
) -> List[Dict[str, object]]:
    """
    Extract named entities from a single text using the provided spaCy model.

    Parameters
    ----------
    text : str
        Input text.
    nlp : spacy.language.Language
        Loaded spaCy model.
    config : NerModelConfig
        NER configuration.

    Returns
    -------
    entities : list of dict
        Each dict contains 'text', 'label', 'start_char', 'end_char'.
    """
    doc = nlp(text)
    entities: List[Dict[str, object]] = []

    for ent in doc.ents:
        if config.entity_types_to_keep is not None and ent.label_ not in config.entity_types_to_keep:
            continue
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            }
        )

    return entities