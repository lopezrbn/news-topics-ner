from __future__ import annotations

from dotenv import load_dotenv

from news_nlp.config import paths
from news_nlp.pipelines.jobs.cli_utils import parse_args, parse_sources_arg
from news_nlp.pipelines.jobs.ner_extractor_inference_job import run_ner_extractor_inference_job


def main() -> None:
    """
    Entry point for the NER extractor inference pipeline.
    """
    # 1) Load environment variables
    load_dotenv(paths.ENV_FILE)

    # 2) Parse CLI arguments
    args = parse_args(module="ner_extractor")
    sources = parse_sources_arg(args.sources)
    mode_ner_extractor = args.mode_ner_extractor  # currently only "incremental" is accepted
    print(f"Mode: {mode_ner_extractor} (only 'incremental' is implemented for NER).")

    # 3) Run NER inference job
    run_ner_extractor_inference_job(
        sources=sources,
        mode_ner_extractor=mode_ner_extractor,
    )


if __name__ == "__main__":
    main()
