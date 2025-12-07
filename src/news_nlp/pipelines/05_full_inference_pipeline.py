from __future__ import annotations

from dotenv import load_dotenv

from news_nlp.config import paths
from news_nlp.pipelines.jobs.cli_utils import parse_args, parse_sources_arg
from news_nlp.pipelines.jobs.full_inference_job import run_full_inference_job


def main() -> None:
    """
    Entry point for the full inference pipeline (topics + NER).
    """
    # 1) Load environment variables
    load_dotenv(paths.ENV_FILE)

    # 2) Parse CLI arguments
    args = parse_args(module="full_inference")
    sources = parse_sources_arg(args.sources)
    mode_topics_detector = args.mode_topics_detector
    mode_ner_extractor = args.mode_ner_extractor
    id_run = args.id_run

    # 3) Run full inference job
    run_full_inference_job(
        sources=sources,
        mode_topics_detector=mode_topics_detector,
        mode_ner_extractor=mode_ner_extractor,
        id_run=id_run,
    )


if __name__ == "__main__":
    main()
