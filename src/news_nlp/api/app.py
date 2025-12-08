from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import RedirectResponse

from news_nlp.api.schemas import (
    TextInput,
    TopicsResponse,
    EntitiesResponse,
    AnalyzeResponse,
    HealthResponse,
)
from news_nlp.api.dependencies import (
    get_topics_service,
    get_ner_service,
    TopicsService,
    NerService,
)


app = FastAPI(
    title="news-topics-ner API",
    description=(
        "Microservice for topic assignment and named entity recognition "
        "on news articles."
    ),
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
async def root():
    """
    Redirect root path to the interactive API docs.
    """
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health(
    topics_service: TopicsService = Depends(get_topics_service),
    ner_service: NerService = Depends(get_ner_service),
) -> HealthResponse:
    """
    Simple health-check endpoint that verifies models are loaded.
    """
    # If we got here, both services were constructed without raising.
    return HealthResponse(
        status="ok",
        topics_model_loaded=True,
        ner_model_loaded=True,
        active_topics_run_id=topics_service.id_run,
        entity_types_supported=ner_service.config.entity_types_to_keep,
    )


@app.post("/v1/topics", response_model=TopicsResponse)
async def infer_topics(
    payload: TextInput,
    topics_service: TopicsService = Depends(get_topics_service),
) -> TopicsResponse:
    """
    Infer topic for a single news text using the active topics model.
    """
    title = payload.title
    text = payload.text
    full_text = f"{title}. {text}" if title else text
    if not full_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'text' must not be empty.",
        )

    topic_info = topics_service.predict_topic_for_text_api(full_text)

    return TopicsResponse(topics=topic_info)


@app.post("/v1/entities", response_model=EntitiesResponse)
async def infer_entities(
    payload: TextInput,
    ner_service: NerService = Depends(get_ner_service),
) -> EntitiesResponse:
    """
    Extract named entities from a single text using the spaCy NER model.
    """
    title = payload.title
    text = payload.text
    full_text = f"{title}. {text}" if title else text
    if not full_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'text' must not be empty.",
        )

    entities = ner_service.extract_entities_for_text_api(full_text)
    return EntitiesResponse(entities=entities)


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze_text(
    payload: TextInput,
    topics_service: TopicsService = Depends(get_topics_service),
    ner_service: NerService = Depends(get_ner_service),
) -> AnalyzeResponse:
    """
    Combined endpoint: infer topics + entities for a single text.
    """
    title = payload.title
    text = payload.text
    full_text = f"{title}. {text}" if title else text
    if not full_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'text' must not be empty.",
        )

    topic_info = topics_service.predict_topic_for_text_api(full_text)
    entities = ner_service.extract_entities_for_text_api(full_text)

    return AnalyzeResponse(
        topics=topic_info,
        entities=entities,
    )
