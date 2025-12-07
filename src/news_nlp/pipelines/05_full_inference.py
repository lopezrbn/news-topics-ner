from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional

from dotenv import load_dotenv

from news_nlp.config import paths
from news_nlp.db.connection import get_engine
from news_nlp.topics_detector.db_io import get_active_run_id
from news_nlp.topics_detector.pipeline_jobs import run_topics_detector_inference_job
from news_nlp.ner_extractor.pipeline_jobs import run_ner_inference_job


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the full inference pipeline.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Full inference pipeline: topics detector + NER extractor.\n"
            "It will first run topics detector inference, then NER extractor inference."
        )
    )

    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help=(
            "Comma-separated list of sources to process "
            "(e.g. 'train', 'train,test', 'prod', 'all'). "
            "Default: 'all'."
        ),
    )

    parser.add_argument(
        "--topics-detector-mode",
        type=str,
        choices=["incremental", "overwrite"],
        default="incremental",
        help=(
            "Mode for topics detector inference: "
            "'incremental' (only news without topics for this run) or "
            "'overwrite' (delete previous assignments for this run+sources, then recompute). "
            "Default: 'incremental'."
        ),
    )

    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help=(
            "Run id (id_run) in topics_model_training_runs for topics inference. "
            "If not provided, the active run (is_active=true) will be used."
        ),
    )

    return parser.parse_args()


def parse_sources_arg(sources: str) -> Optional[List[str]]:
    """
    Parse a comma-separated sources string.

    Examples
    --------
    "train"      -> ["train"]
    "train, test" -> ["train", "test"]
    "all"        -> None  (meaning: no filtering by source)

    Returns
    -------
    list[str] or None
        List of sources to filter by, or None for all sources.
    """
    sources = sources.strip().lower()
    if sources == "all":
        return None

    parts = [s.strip() for s in sources.split(",") if s.strip()]
    if not parts:
        return None

    return parts


def run_full_inference_job(
    sources: Optional[List[str]],
    topics_detector_mode: str,
    run_id: Optional[int],
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
    if run_id is None:
        run_id = get_active_run_id(engine=engine)
        print(f"No run-id provided. Using active topics run id_run={run_id}.")
    else:
        print(f"Using provided topics run id_run={run_id}.")

    # 2) Run topics inference job
    print("Starting topics inference job...")
    run_topics_detector_inference_job(
        id_run=run_id,
        sources=sources,
        mode=topics_detector_mode,
    )
    print("Topics inference job finished.")

    # 3) Run NER inference job (always incremental)
    print("Starting NER inference job (incremental)...")
    run_ner_inference_job(
        sources=sources,
    )
    print("NER inference job finished.")

    print("Full inference job (topics + NER) completed successfully.")


def main() -> None:
    """
    Entry point for the full inference pipeline (topics + NER).
    """
    # 1) Load environment variables
    load_dotenv(paths.ENV_FILE)

    # 2) Parse CLI arguments
    args = parse_args()
    sources = parse_sources_arg(args.sources)
    topics_detector_mode = args.topics_detector_mode
    run_id = args.run_id

    # 3) Run full inference job
    run_full_inference_job(
        sources=sources,
        topics_detector_mode=topics_detector_mode,
        run_id=run_id,
    )


if __name__ == "__main__":
    main()
