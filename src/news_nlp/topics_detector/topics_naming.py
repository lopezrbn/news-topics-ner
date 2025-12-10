import json
import os
from typing import Dict, List, Tuple
import yaml
from openai import OpenAI

from news_nlp.config import paths


def load_prompt_from_yaml(path: str, module: str) -> str:
	"""
	Load the prompt for the given module from a YAML file.
	Args:
		path (str): Path to the YAML file.
		module (str): Module name to get the prompt for.
	Returns:
		str: The prompt string.
	"""
	with open(path, 'r') as file:
		prompts = yaml.safe_load(file)
	return prompts.get(module, "")


def infer_llm(prompt: str, system_prompt: str, model: str, api_key: str) -> str:
    """
    Call an LLM model with system and user prompts and return the text response.
    """
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=prompt,
    )

    # Extract first text output (simplest case)
    return response.output_text


def generate_topic_names_with_llm(
    top_terms_per_topic: Dict[int, List[Tuple[str, float]]],
) -> Dict[int, str]:
    """
    Use an LLM (via Llms_inferer) to generate human-readable names
    for each topic based on its top terms.

    Returns a dict:
      id_topic -> topic_name
    """
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY", None)
    if api_key is None:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    
    # Prepare input: id_topic -> list of terms (strings)
    topics_payload: Dict[str, List[str]] = {
        str(id_topic): [term for term, _ in terms]
        for id_topic, terms in top_terms_per_topic.items()
    }
    user_prompt = json.dumps(topics_payload)

    system_prompt=load_prompt_from_yaml(paths.PROMPTS_FILE, module="topics_namer")

    # Infer with LLM
    response = infer_llm(
        model="gpt-5",
        system_prompt=system_prompt,
        prompt=user_prompt,
        api_key=api_key,
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