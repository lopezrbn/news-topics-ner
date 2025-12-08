from __future__ import annotations

from dotenv import load_dotenv

from news_nlp.config import paths
from news_nlp.pipelines.jobs.cli_utils import parse_args, parse_sources_arg
from news_nlp.db.connection import get_engine
from news_nlp.topics_detector.db_io import get_active_id_run
from news_nlp.pipelines.jobs.topics_detector_inference_job import run_topics_detector_inference_job


def main() -> None:
    """
    Entry point for the topics detector inference pipeline.
    """
    # Load environment variables
    load_dotenv(paths.ENV_FILE)

    # Parse command line arguments
    args = parse_args(module="topics_detector")
    sources = parse_sources_arg(args.sources)
    mode_topics_detector = args.mode_topics_detector
    id_run = args.id_run

    # Get DB engine
    engine = get_engine()

    if id_run is None:
        id_run = get_active_id_run(engine=engine)
        print(f"No run-id provided. Using active run id_run={id_run}.")
    else:
        print(f"Using provided run id_run={id_run}.")

    run_topics_detector_inference_job(
        sources=sources,
        mode_topics_detector=mode_topics_detector,
        id_run=id_run,
    )


if __name__ == "__main__":
    main()
