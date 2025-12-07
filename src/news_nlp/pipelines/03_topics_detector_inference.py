from __future__ import annotations

import argparse
from typing import List, Optional

from dotenv import load_dotenv

from news_nlp.config import paths
from news_nlp.db.connection import get_engine
from news_nlp.topics_detector.db_io import get_active_run_id
from news_nlp.topics_detector.pipeline_jobs import run_topics_detector_inference_job


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the topics inference pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Topics detector inference pipeline (parametrizable by sources, mode, run_id)."
    )

    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help=(
            "Comma-separated list of sources to process "
            "(e.g. 'train', 'train, test', 'prod', 'all'). "
            "Default: 'all'."
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["incremental", "overwrite"],
        default="incremental",
        help=(
            "Inference mode: "
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
            "Run id (id_run) in topics_model_training_runs. "
            "If not provided, the active run (is_active=true) will be used."
        ),
    )

    return parser.parse_args()


def parse_sources_arg(sources: str) -> Optional[List[str]]:
    """
    Parse a comma-separated sources string.

    Examples
    --------
    "train"        -> ["train"]
    "train, test"   -> ["train", "test"]
    "all"          -> None  (meaning: no filtering by source)

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


def main() -> None:
    """
    Entry point for the topics detector inference pipeline.
    """
    # Load environment variables
    load_dotenv(paths.ENV_FILE)

    # Parse command line arguments
    args = parse_args()
    sources = parse_sources_arg(args.sources)
    mode = args.mode
    run_id = args.run_id

    # Get DB engine
    engine = get_engine()

    if run_id is None:
        run_id = get_active_run_id(engine=engine)
        print(f"No run-id provided. Using active run id_run={run_id}.")
    else:
        print(f"Using provided run id_run={run_id}.")

    run_topics_detector_inference_job(
        id_run=run_id,
        sources=sources,
        mode=mode,
    )


if __name__ == "__main__":
    main()
