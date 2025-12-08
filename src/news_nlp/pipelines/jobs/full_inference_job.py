from __future__ import annotations

from typing import Optional, Iterable, Literal

from news_nlp.db.connection import get_engine
from news_nlp.topics_detector.db_io import get_active_id_run
from news_nlp.pipelines.jobs.topics_detector_inference_job import run_topics_detector_inference_job
from news_nlp.pipelines.jobs.ner_extractor_inference_job import run_ner_extractor_inference_job


def run_full_inference_job(
    sources: Optional[Iterable[str]] = None,
    mode_topics_detector: Optional[Literal["incremental", "overwrite"]] = "incremental",
    mode_ner_extractor: Optional[Literal["incremental"]] = "incremental",
    id_run: Optional[int] = None,
) -> None:
    """
    Run full inference job: topics detector inference + NER extractor inference.

    Steps:
      1) Resolve id_run for topics (active run if not provided).
      2) Run topics detector inference job for that run and sources.
      3) Run NER extractor inference job for the same sources (incremental mode).
    """
    engine = get_engine()

    # 1) Resolve run_id for topics
    if id_run is None:
        id_run = get_active_id_run(engine=engine)
        print(f"No run-id provided. Using active topics run id_run={id_run}.")
    else:
        print(f"Using provided topics run id_run={id_run}.")

    # 2) Run topics inference job
    print("Starting topics inference job...")
    run_topics_detector_inference_job(
        sources=sources,
        mode_topics_detector=mode_topics_detector,
        id_run=id_run,
    )
    print("Topics inference job finished.")

    # 3) Run NER inference job (always incremental)
    print("Starting NER inference job (incremental)...")
    run_ner_extractor_inference_job(
        sources=sources,
        mode_ner_extractor=mode_ner_extractor,
    )
    print("NER inference job finished.")

    print("Full inference job (topics + NER) completed successfully.")
