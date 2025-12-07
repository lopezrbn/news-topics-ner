# src/news_nlp/pipelines/jobs/cli_utils.py

from __future__ import annotations

import argparse
from typing import Iterable, List, Optional, Literal


def parse_args(
    module: Literal["topics_detector", "ner_extractor", "full_inference"] = "full_inference",
    description: Optional[str] = None,
) -> argparse.Namespace:
    """
    Build an ArgumentParser for inference pipelines.

    It includes:
      --sources
      --mode_topics_detector // only for topics_detector and full_inference
      --mode_ner_extractor   // only for ner_extractor and full_inference
      --id_run               // only for topics_detector and full_inference

    Parameters:
    - module : Literal["topics_detector", "ner_extractor", "full_inference"]
        The module for which to build the parser.
    - description : Optional[str]
        Custom description for the ArgumentParser. If None, a default description will be used.

    Returns:
    - argparse.Namespace
        The parsed arguments.
    """

    # Determine default description if not provided
    if description is None:

        if module == "topics_detector":
            description = "Topics detector inference pipeline (parametrizable by sources, mode, run_id)."

        elif module == "ner_extractor":
            description = "NER extractor inference pipeline (incremental, parametrizable by sources)."

        elif module == "full_inference":
            description = (
                "Full inference pipeline: topics detector + NER extractor.\n"
                "It will first run topics detector inference, then NER extractor inference."
            )

        else:
            raise ValueError(f"Unknown module: {module}")

    # Build parser
    parser = argparse.ArgumentParser(description=description)

    # Add common argument --sources
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

    # Add module-specific arguments for topics_detector and full_inference
    if module == "topics_detector" or module == "full_inference":

        parser.add_argument(
            "--id-run",
            type=int,
            default=None,
            help=(
                "Run id (id_run) in topics_model_training_runs. "
                "If not provided, the active run (is_active=true) will be used."
            ),
        )

        parser.add_argument(
            "--mode-topics-detector",
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

    # Add module-specific arguments for ner_extractor and full_inference
    if module == "ner_extractor" or module == "full_inference":

        parser.add_argument(
            "--mode-ner-extractor",
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
