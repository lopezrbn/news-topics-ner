from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv

from news_nlp.config import paths
from news_nlp.db.connection import get_engine
from news_nlp.topics_detector.db_io import get_active_id_run, get_topics_metadata_dict
from news_nlp.topics_detector.inference import predict_topic_for_text
from news_nlp.topics_detector.model import load_topic_detector_pipeline
from news_nlp.ner_extractor.model import NerModelConfig, extract_entities_for_text, load_spacy_model


load_dotenv(paths.ENV_FILE)


class TopicsService:
    """
    Service responsible for topics-related inference and metadata.
    It loads the active topics detector model run and the corresponding sklearn pipeline.
    """

    def __init__(self) -> None:
        self.engine = get_engine()
        self.id_run: int = get_active_id_run(self.engine)
        self.pipeline = load_topic_detector_pipeline(id_run=self.id_run)
        self.topics_metadata = get_topics_metadata_dict(engine=self.engine, id_run=self.id_run)


    def predict_topic_for_text_api(self, text: str) -> Tuple[int, Dict[str, object]]:
        """
        Predict the topic for a single text, returning topic id and metadata.
        Parameters
        ----------
        text : str
            Raw news text.
        Returns
        -------
        result : dict
            Keys:
              - "topic_id": int
              - "topic_name": Optional[str]
              - "top_terms": List[str]
        """
        id_topic = predict_topic_for_text(
            text=text,
            pipeline=self.pipeline,
            apply_cleaning=True,
        )

        result: Dict[str, object] = {
            "id_run": self.id_run,
            "id_topic": id_topic,
            "topic_name": self.topics_metadata.get(id_topic, {}).get("topic_name", f"Topic {id_topic}"),
            "top_terms": self.topics_metadata.get(id_topic, {}).get("top_terms", []),
        }

        return result


class NerService:
    """
    Service responsible for named entity recognition (NER) inference.
    It loads a spaCy NER model according to the provided configuration.
    """

    def __init__(
        self,
        config: NerModelConfig = None,
    ) -> None:

        self.config = config or NerModelConfig(entity_types_to_keep=["PERSON", "ORG", "GPE", "LOC"])
        self.nlp = load_spacy_model(self.config)

    def extract_entities_for_text_api(
        self,
        text: str,
    ) -> List[Dict[str, object]]:
        """
        Extract named entities from a single text.
        """
        return extract_entities_for_text(
            text=text,
            nlp=self.nlp,
            config=self.config,
        )


# Singleton-style instances reused across requests
_topics_service: Optional[TopicsService] = None
_ner_service: Optional[NerService] = None


def get_topics_service() -> TopicsService:
    """
    Return a singleton TopicsService instance, loading it lazily on first use.
    """
    global _topics_service
    if _topics_service is None:
        _topics_service = TopicsService()
    return _topics_service


def get_ner_service() -> NerService:
    """
    Return a singleton NerService instance, loading it lazily on first use.
    """
    global _ner_service
    if _ner_service is None:
        _ner_service = NerService()
    return _ner_service
