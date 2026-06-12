import os
import random
import re
import json
from typing import List, Optional

_METRICS_DIR = os.path.dirname(os.path.abspath(__file__))
GPT_KEY_FILE = os.path.join(_METRICS_DIR, 'gpt_key.txt')

# Evaluator LLM (judge): openai (remote GPT) | ollama | vllm
#   EVAL_BACKEND=ollama
#   EVAL_BASE_URL=http://127.0.0.1:11434/v1
#   EVAL_API_KEY=ollama
#   EVAL_MODEL=qwen2.5:14b-instruct
EVAL_BACKEND = os.environ.get('EVAL_BACKEND', 'openai').strip().lower()
EVAL_BASE_URL = os.environ.get('EVAL_BASE_URL', 'https://xiaoai.plus/v1').strip()
EVAL_API_KEY = os.environ.get('EVAL_API_KEY', '').strip()
EVAL_MODEL_DEFAULT = os.environ.get('EVAL_MODEL', 'gpt-4o-2024-11-20').strip()
EVAL_TEMPERATURE = float(os.environ.get('EVAL_TEMPERATURE', '0'))


def get_eval_model(explicit_model: Optional[str] = None) -> str:
    if explicit_model:
        return explicit_model
    return EVAL_MODEL_DEFAULT


def _resolve_api_key() -> str:
    if EVAL_API_KEY:
        return EVAL_API_KEY
    if EVAL_BACKEND in ('ollama', 'vllm', 'local'):
        return 'local'
    with open(GPT_KEY_FILE, 'r', encoding='utf-8') as f:
        api_keys = [line.strip() for line in f.readlines() if line.strip()]
    if not api_keys:
        raise RuntimeError(
            f'No API key in {GPT_KEY_FILE}. Set EVAL_BACKEND=ollama/vllm for local Qwen, '
            'or add keys for remote GPT.'
        )
    return random.choice(api_keys)


def get_eval_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            'openai is required for remote/API judges. Install with: pip install openai '
            'Or use --judge gemma-local for in-process Gemma scoring.'
        ) from exc
    return OpenAI(base_url=EVAL_BASE_URL, api_key=_resolve_api_key())


def workflow(model_name, instruction, input_text):
    """Execute a single API call to the configured evaluator LLM."""
    client = get_eval_client()
    model = get_eval_model(model_name)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
        ],
        temperature=EVAL_TEMPERATURE,
    )
    return completion.choices[0].message.content


def workflow_multi_turn(model_name, input_text, history_messages):
    client = get_eval_client()
    model = get_eval_model(model_name)

    history_messages.append({"role": "user", "content": input_text})
    completion = client.chat.completions.create(
        model=model,
        messages=history_messages,
        temperature=EVAL_TEMPERATURE,
    )
    return completion.choices[0].message.content


def load_instruction(file_path):
    """Load instruction text from file."""
    with open(file_path, 'r', encoding='utf-8') as fp:
        return fp.read()


def safe_json_parse(model_output, retry_count=0):
    """Safely parse JSON and handle formatting errors."""
    max_retries = 3
    if retry_count >= max_retries:
        print("JSON parse error after maximum retries")
        return None
    try:
        parsed_output = json.loads(model_output)
        return parsed_output
    except json.JSONDecodeError as e:
        corrected_output = request_correction_from_model(model_output, str(e), retry_count)
        return safe_json_parse(corrected_output, retry_count + 1)


def request_correction_from_model(
    incorrect_output,
    error_message,
    retry_count,
    model_name=None,
):
    """Request model to fix JSON formatting errors."""
    max_retries = 3
    if retry_count >= max_retries:
        return incorrect_output

    system_prompt = 'You are a JSON format modifier.'
    input_text = (
        f"Fixed the following output JSON format error, ensure that it is a valid JSON string, "
        f"and the current error message is{error_message}"
        f"only output the correct JSON string that can be parsed, do not output other content:\n"
        f"{incorrect_output}"
    )

    corrected_completion = workflow(
        model_name=model_name,
        instruction=system_prompt,
        input_text=input_text,
    ).replace('```json', '').replace('```', '').strip()

    print(f'Try correct {retry_count}\n before:\n{incorrect_output}\nafter:\n{corrected_completion}')

    try:
        output = json.loads(corrected_completion)
        return json.dumps(output)
    except json.JSONDecodeError as e:
        return request_correction_from_model(corrected_completion, str(e), retry_count + 1)
