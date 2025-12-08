from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class TextInput(BaseModel):
    """
    Input payload for endpoints that receive one news text.
    """
    title: Optional[str] = Field(
        default=None,
        description="Optional title of the news article.",
    )
    text: str = Field(
        ...,
        description="Raw text of the news article.",
        min_length=1,
    )


class TopicInfo(BaseModel):
    """
    Topic information returned by the topics endpoint.
    """
    id_run: int = Field(..., description="ID of the active topics detector model run.")
    id_topic: int = Field(..., description="Cluster/topic ID assigned to the text.")
    topic_name: Optional[str] = Field(
        default=None,
        description="Human-readable topic name, if available.",
    )
    top_terms: List[str] = Field(
        default_factory=list,
        description="Top-n most representative terms associated with this topic.",
    )
    # Not used for now
    # model_name: Optional[str] = Field(
    #     default=None,
    #     description="Name of the topics detector model.",
    # )
    # silhouette_score: Optional[float] = Field(
    #     default=None,
    #     description="Silhouette score of the topics detector model (if available).",
    # )


class TopicsResponse(BaseModel):
    """
    Response schema for /v1/topics endpoint.
    """
    topics: TopicInfo


class Entity(BaseModel):
    """
    One named entity found in the text.
    """
    text: str = Field(..., description="Entity surface form.")
    label: str = Field(..., description="Entity type label (e.g. PERSON, ORG, LOC).")
    start_char: int = Field(..., description="Start character index in the text.")
    end_char: int = Field(..., description="End character index in the text.")


class EntitiesResponse(BaseModel):
    """
    Response schema for /v1/entities endpoint.
    """
    entities: List[Entity]


class AnalyzeResponse(BaseModel):
    """
    Response schema for /v1/analyze endpoint (topics detector + entities extractor).
    """
    topics: TopicInfo
    entities: List[Entity]


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """
    status: Literal["ok", "error"] = "ok"
    topics_model_loaded: bool
    ner_model_loaded: bool
    active_topics_run_id: Optional[int] = None
    entity_types_supported: Optional[List[str]] = None
    error_message: Optional[str] = None
