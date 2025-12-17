import json
import os
from typing import Dict, List, Tuple
import yaml
from openai import OpenAI
import pandas as pd

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
	df_train: pd.DataFrame
) -> Dict[int, str]:
	"""
	Use an LLM (via Llms_inferer) to generate human-readable names
	for each topic based on its top terms.

	Returns a dict:
	  id_topic -> topic_name
	"""
	
	MAX_LEN_TEXT = 7500
	
	# Get API key
	api_key = os.getenv("OPENAI_API_KEY", None)
	if api_key is None:
		raise ValueError("OPENAI_API_KEY is not set in the environment.")
	
	# Load system prompt
	system_prompt=load_prompt_from_yaml(paths.PROMPTS_FILE, module="topics_namer_2")

	# Select the top 3 representative articles per topic (closest to centroid)
	df_best_texts = (
		df_train[["id_topic", "text", "len_text", "dist_centroid"]]
		.sort_values(["id_topic", "dist_centroid"], ascending=[True, True])
		.groupby("id_topic", as_index=False)
		.head(3)
		.copy()
	)

	# Truncate texts to fit within token limits
	df_best_texts["text"] = df_best_texts["text"].str.slice(0, MAX_LEN_TEXT)
	df_best_texts["len_text"] = df_best_texts["text"].str.len()
	
	# Build a dict mapping topic_id to list of representative texts
	texts_by_topic = (
		df_best_texts
		.groupby("id_topic")["text"]
		.apply(list)
		.to_dict()
	)

	topic_names_raw = {}
	for id_topic, texts_list in texts_by_topic.items():
		user_prompt_obj = {
			str(id_topic): {
				"terms": top_terms_per_topic,
				"news_texts": texts_list,
			}
		}
		user_prompt_json = json.dumps(user_prompt_obj, ensure_ascii=False, indent=2)
		
		print(f"Generating name for topic {id_topic}...")
		response = infer_llm(
			model="gpt-5.2",
			system_prompt=system_prompt,
			prompt=user_prompt_json,
			api_key=os.getenv("OPENAI_API_KEY", None),
		)

		# Parse the response JSON
		try:
			response_json = json.loads(response)
			print(f"\tTopic {id_topic} name: {response_json[str(id_topic)]}")
		except json.JSONDecodeError as exc:
			with open(paths.TOPIC_NAMES_FILE, "r") as f:
				response_json = json.load(f)
		
		topic_names_raw.update(response_json)

		with open(paths.TOPIC_NAMES_FILE, "w", encoding="utf-8") as f:
			json.dump(topic_names, f, ensure_ascii=False, indent=4)

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