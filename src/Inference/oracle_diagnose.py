import os
import re
import json
import time
import random
import concurrent.futures
import numpy as np
import requests
from tqdm import tqdm
from openai import OpenAI

# Configuration Constants
# Change these to your own API keys and model URLs
OPENAI_COMPAT_BASE_URL = 'https://xiaoai.plus/v1'
OPENAI_COMPAT_API_KEY = 'sk-0KdA7qGS7hq6k2ET6PFDKiIecav56NHlY5lPwfnrtC3WdeHk'

QWQ_URL = OPENAI_COMPAT_BASE_URL
QWQ_API_KEY = OPENAI_COMPAT_API_KEY

GEMINI_URL = OPENAI_COMPAT_BASE_URL
GEMINI_API_KEY = OPENAI_COMPAT_API_KEY

# Full URL only for raw requests.post (not for OpenAI client)
DEEPSEEK_R1_URL = f'{OPENAI_COMPAT_BASE_URL}/chat/completions'
DEEPSEEK_R1_API_KEY = OPENAI_COMPAT_API_KEY

O1_API_KEY_LIST = [
    'sk-0KdA7qGS7hq6k2ET6PFDKiIecav56NHlY5lPwfnrtC3WdeHk'
]

DEFAULT_SYSTEM_PROMPT = "You are a professional doctor"
DATA_PATH = '../../data/MedRBench/test_cases.json'
PROMPT_TEMPLATE_PATH = './instructions/oracle_diagnose.txt'

# Local Qwen3 via vLLM (override with env vars on the server)
QWEN3_URL = os.environ.get('QWEN3_URL', 'http://127.0.0.1:8000/v1')
QWEN3_API_KEY = os.environ.get('QWEN3_API_KEY', 'EMPTY')
QWEN3_MODEL = os.environ.get('QWEN3_MODEL', 'Qwen/Qwen3-8B-Instruct')
QWEN3_MODEL_KEY = os.environ.get('QWEN3_MODEL_KEY', 'qwen3-8b')
# transformers (P100 / no vLLM) or vllm (OpenAI API on port 8000)
QWEN3_BACKEND = os.environ.get('QWEN3_BACKEND', 'transformers')

# =====================
# MODEL API INTERFACES
# =====================

def create_qwq_client():
    """Create and return OpenAI client for QwQ model"""
    return OpenAI(
        base_url=QWQ_URL,
        api_key=QWQ_API_KEY
    )

def query_qwq_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the QwQ model and handle potential rate limit errors"""
    client = create_qwq_client()
    while True:
        try:
            response = client.chat.completions.create(
                model="Qwen/QwQ-32B-Preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                stream=False, 
                max_tokens=4096
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            error_message = str(e)
            if '429' in error_message:
                print(f"Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
                print('Retrying...')
            else:
                print(f"Error querying QwQ model: {e}")
                return None

def query_llama3_r1_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the Llama3-R1 model with reasoning support"""
    client = create_qwq_client()
    while True:
        try:
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                stream=False, 
                max_tokens=4096
            )
            content = response.choices[0].message.content
            reasoning_content = response.choices[0].message.reasoning_content
            return content, reasoning_content
        except Exception as e:
            error_message = str(e)
            if '429' in error_message:
                print(f"Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
                print('Retrying...')
            else:
                print(f"Error querying Llama3-R1 model: {e}")
                return None, None

def query_deepseek_r1_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query DeepSeek-R1 model via direct HTTP request"""
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-r1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ],
        "temperature": 0.6,
        "stream": False,
        "max_tokens": 10000,
    }
    max_retry = 3
    curr_retry = 0
    
    while curr_retry < max_retry:
        try:
            response = requests.post(DEEPSEEK_R1_URL, json=data, headers=headers)
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            reasoning = content.split('</think>')[0].replace('<think>', '').strip()
            answer = content.split('</think>')[1].strip()
            return reasoning, answer
        except Exception as e:
            curr_retry += 1
            print(f"Error: {e}, retrying... {curr_retry}/{max_retry}")
            time.sleep(2)
    
    return None, None

def query_o1_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the O1/O3-mini model with round-robin API key selection"""
    max_try_times = 3
    curr_try = 0
    
    while curr_try < max_try_times:
        try:
            client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key=random.choice(O1_API_KEY_LIST)
            )
            completion = client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            curr_try += 1
            if curr_try >= max_try_times:
                print(f"Error querying O1 model: {e}")
                return "Error."
            time.sleep(5)

def query_qwen3_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query local Qwen3 model via vLLM OpenAI-compatible API."""
    client = OpenAI(base_url=QWEN3_URL, api_key=QWEN3_API_KEY)
    max_retry = 3
    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model=QWEN3_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text},
                ],
                max_tokens=4096,
                temperature=0.6,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying Qwen3 model (attempt {attempt + 1}/{max_retry}): {e}")
            time.sleep(5)
    return None

def query_gemini_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the Gemini model and handle rate limits"""
    client = OpenAI(
        base_url=GEMINI_URL,
        api_key=GEMINI_API_KEY
    )
    while True:
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-flash-thinking",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                stream=False, 
                max_tokens=4096
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            error_message = str(e)
            if '429' in error_message:
                print(f"Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
                print('Retrying...')
            else:
                print(f"Error querying Gemini model: {e}")
                return None

def query_baichuan_model(input_text, model, tokenizer, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the Baichuan model using transformers library"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate text
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated text
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        print(f"Error querying Baichuan model: {e}")
        return None

# =====================
# DATA PROCESSING
# =====================

def load_instruction(template_path):
    """Load instruction template from file"""
    try:
        with open(template_path, encoding='utf-8') as fp:
            return fp.read()
    except Exception as e:
        print(f"Error loading instruction template: {e}")
        return None

def load_data(data_path):
    """Load and parse case data from JSON file"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}

def save_results(data, filename):
    """Save results to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

# =====================
# INFERENCE PROCESSORS
# =====================

def process_qwq_data(data_id, data, prompt_template):
    """Process a single data item with QwQ model"""   
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        result['input'] = prompt
        
        response = query_qwq_model(prompt)
        result['content'] = response
        
        return data_id, result
    except Exception as e:
        print(f"Error processing QwQ data {data_id}: {e}")
        return data_id, None

def process_deepseek_r1_data(data_id, data, prompt_template):
    """Process a single data item with DeepSeek-R1 model"""
    
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        result['input'] = prompt
        
        reasoning, answer = query_deepseek_r1_model(prompt)
        if reasoning and answer:
            result['out_reasoning'] = reasoning
            result['out_answer'] = answer
                        
            return data_id, result
        else:
            return data_id, None
    except Exception as e:
        print(f"Error processing DeepSeek-R1 data {data_id}: {e}")
        return data_id, None

def process_o1_data(data_id, data, prompt_template):
    """Process a single data item with O1/O3-mini model"""
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        result['input'] = prompt
        
        response = query_o1_model(prompt)
        result['content'] = response
                
        return data_id, result
    except Exception as e:
        print(f"Error processing O1 data {data_id}: {e}")
        return data_id, None

def process_qwen3_data(data_id, data, prompt_template):
    """Process a single data item with local Qwen3 model."""
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        result['input'] = prompt

        response = query_qwen3_model(prompt)
        result['content'] = response

        return data_id, result
    except Exception as e:
        print(f"Error processing Qwen3 data {data_id}: {e}")
        return data_id, None

def process_gemini_data(data_id, data, prompt_template):
    """Process a single data item with Gemini model"""
       
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        result['input'] = prompt
        
        response = query_gemini_model(prompt)
        result['content'] = response
                
        return data_id, result
    except Exception as e:
        print(f"Error processing Gemini data {data_id}: {e}")
        return data_id, None

# =====================
# MAIN INFERENCE RUNNERS
# =====================

def run_inference_with_model(
    process_func, 
    model_name, 
    output_filename, 
    max_workers=8
):
    """Generic function to run inference with any model"""
    print(f"Running inference with {model_name} model")
    
    prompt_template = load_instruction(PROMPT_TEMPLATE_PATH)
    data = load_data(DATA_PATH)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for data_id, data_item in data.items():
            futures.append(executor.submit(process_func, data_id, data_item, prompt_template))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing with {model_name}"):
            data_id, result = future.result()
            if result is not None:
                data[data_id][model_name.lower()] = result
    save_results(data, output_filename)

def inference_qwq():
    """Run inference with QwQ model"""
    run_inference_with_model(
        process_qwq_data,
        "qwq",
        "oracle_diagnosis_qwq.json",
        max_workers=8
    )

def inference_deepseek_r1():
    """Run inference with DeepSeek-R1 model"""
    run_inference_with_model(
        process_deepseek_r1_data,
        "deepseekr1",
        "oracle_diagnosis_deepseekr1.json",
        max_workers=8
    )

def inference_o1():
    """Run inference with O1/O3-mini model"""
    run_inference_with_model(
        process_o1_data,
        "o3-mini",
        "oracle_diagnosis_o3-mini.json",
        max_workers=8
    )

def inference_gemini():
    """Run inference with Gemini model"""
    run_inference_with_model(
        process_gemini_data,
        "gemini2-ft",
        "oracle_diagnosis_gemini.json",
        max_workers=8
    )

def inference_qwen3():
    """Run inference with local Qwen3 model (vLLM). Use max_workers=1 for single GPU."""
    run_inference_with_model(
        process_qwen3_data,
        QWEN3_MODEL_KEY,
        "oracle_diagnosis_qwen3.json",
        max_workers=1,
    )

def inference_qwen3_transformers():
    """Run Qwen3 with HuggingFace transformers (for P100 / servers without vLLM)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Running inference with Qwen3 (transformers): {QWEN3_MODEL}")
    prompt_template = load_instruction(PROMPT_TEMPLATE_PATH)
    data = load_data(DATA_PATH)

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN3_MODEL,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    for data_id in tqdm(data.keys(), desc=f"Processing with {QWEN3_MODEL_KEY}"):
        patient_case = data[data_id]['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)

        response = query_baichuan_model(prompt, model, tokenizer)
        if response:
            data[data_id][QWEN3_MODEL_KEY] = {
                'input': prompt,
                'content': response,
            }

    save_results(data, "oracle_diagnosis_qwen3.json")

def inference_baichuan():
    """Run inference with Baichuan model directly (not using concurrency)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("Running inference with Baichuan model")
    prompt_template = load_instruction(PROMPT_TEMPLATE_PATH)
    data = load_data(DATA_PATH)
    
    model_name = "baichuan-inc/Baichuan-M1-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        attn_implementation='sdpa',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    
    for data_id in tqdm(data.keys(), desc="Processing with Baichuan"):
        patient_case = data[data_id]['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        
        response = query_baichuan_model(prompt, model, tokenizer)
        if response:
            data[data_id]['baichuan'] = response
    
    save_results(data, "oracle_diagnosis_baichuan.json")

if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "gemini":
        inference_gemini()
    elif cmd == "qwen3":
        if QWEN3_BACKEND == "vllm":
            inference_qwen3()
        else:
            inference_qwen3_transformers()
    elif cmd:
        print(f"Unknown command: {cmd}. Use: gemini | qwen3")
    else:
        print("Usage: python oracle_diagnose.py gemini")
        print("  Or:  python scripts/inference/run_gemini_oracle_35.py [--force]")