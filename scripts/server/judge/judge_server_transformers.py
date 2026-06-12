#!/usr/bin/env python3
"""
OpenAI-compatible /v1/chat/completions for MedRBench evaluation.
Use when Ollama/vLLM are unavailable (e.g. P100 + no ollama binary).

  conda activate medrbench
  pip install torch transformers accelerate bitsandbytes fastapi uvicorn
  python scripts/server/judge/judge_server_transformers.py --model Qwen/Qwen2.5-7B-Instruct --port 8000

Then:
  export EVAL_BACKEND=vllm
  export EVAL_BASE_URL=http://127.0.0.1:8000/v1
  export EVAL_MODEL=Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = FastAPI()
_model = None
_tokenizer = None
_model_id = ""
_gen_seed: int | None = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.0
    max_tokens: Optional[int] = Field(default=1024, alias="max_tokens")


def load_model(model_id: str, four_bit: bool = True) -> None:
    global _model, _tokenizer, _model_id
    import torch as _torch

    tv = __import__("transformers").__version__
    print(f"transformers={tv}, torch={_torch.__version__}, cuda={_torch.cuda.is_available()}")
    print(f"Loading {model_id} (4bit={four_bit})...")
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"

    _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    _model.eval()
    _model_id = model_id
    print("Model ready.")


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": _model_id, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if _model is None or _tokenizer is None:
        return {"error": "model not loaded"}, 503

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    max_new = req.max_tokens or 1024
    temp = max(req.temperature, 0.0)
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new,
        "do_sample": temp > 0,
        "pad_token_id": _tokenizer.eos_token_id,
    }
    if temp > 0:
        gen_kwargs["temperature"] = temp

    if _gen_seed is not None:
        _torch = torch
        _torch.manual_seed(_gen_seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(_gen_seed)

    t0 = time.time()
    with torch.no_grad():
        out = _model.generate(**inputs, **gen_kwargs)
    text = _tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(t0),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text.strip()},
                "finish_reason": "stop",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-4bit", action="store_true", help="Full fp16 (needs more VRAM)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed before each generation (for temp>0 reproducibility tests)",
    )
    args = parser.parse_args()

    global _gen_seed
    _gen_seed = args.seed
    if _gen_seed is not None:
        torch.manual_seed(_gen_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_gen_seed)
        print(f"Generation seed={_gen_seed}")

    load_model(args.model, four_bit=not args.no_4bit)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
