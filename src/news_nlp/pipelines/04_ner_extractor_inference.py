from __future__ import annotations

import argparse
from typing import List, Optional
from dotenv import load_dotenv

from news_nlp.config import paths
from news_nlp.ner_extractor.pipeline_jobs import run_ner_inference_job


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the NER inference pipeline.
    """
    parser = argparse.ArgumentParser(
        description="NER extractor inference pipeline (incremental, parametrizable by sources)."
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

    # Only incremental mode is supported for NER, to keep things simple for now
    parser.add_argument(
        "--mode",
        type=str,
        choices=["incremental"],
        default="incremental",
        help=(
            "Inference mode. "
            "Currently only 'incremental' is supported for NER: "
            "process news that do not yet have entities_per_news rows."
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


def main() -> None:
    """
    Entry point for the NER extractor inference pipeline.
    """
    # 1) Load environment variables
    load_dotenv(paths.ENV_FILE)

    # 2) Parse CLI arguments
    args = parse_args()
    sources = parse_sources_arg(args.sources)
    mode = args.mode  # currently only "incremental" is accepted
    print(f"Mode: {mode} (only 'incremental' is implemented for NER).")

    # 3) Run NER inference job
    run_ner_inference_job(sources=sources)


if __name__ == "__main__":
    main()
