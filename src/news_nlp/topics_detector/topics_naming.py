import json
import os
from typing import Dict, List, Tuple

from llms_inferer import Llms_inferer

from news_nlp.config import paths


def generate_topic_names_with_llm(
    top_terms_per_topic: Dict[int, List[Tuple[str, float]]],
) -> Dict[int, str]:
    """
    Use an LLM (via Llms_inferer) to generate human-readable names
    for each topic based on its top terms.

    Returns a dict:
      id_topic -> topic_name
    """
    # Prepare input: id_topic -> list of terms (strings)
    topics_payload = {
        str(id_topic): [term for term, _ in terms]
        for id_topic, terms in top_terms_per_topic.items()
    }

    # Initialize Llms_inferer
    api_key = os.getenv("OPENAI_API_KEY", None)
    if api_key is None:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    inferer = Llms_inferer(
        model="gpt-5",
        prompts_path=str(paths.PROMPTS_FILE),
        api_key=api_key,
        run_local=False,
    )

    system_prompt = inferer.get_prompt(module="topics_namer")
    user_prompt = json.dumps(topics_payload)

    response = inferer.infer(
        system_prompt=system_prompt,
        prompt=user_prompt,
    )

    # Response is expected to be a JSON mapping id_topic (string) -> name (string)
    try:
        topic_names_raw = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {response}") from exc

    # Convert keys to int and ensure we return a dict[int, str]
    topic_names: Dict[int, str] = {}
    for key_str, name in topic_names_raw.items():
        try:
            id_topic = int(key_str)
        except ValueError:
            # Skip invalid keys
            continue
        topic_names[id_topic] = str(name)

    return topic_names
