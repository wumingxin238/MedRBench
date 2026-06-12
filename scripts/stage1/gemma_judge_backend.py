"""In-process Gemma instruct model as MedRBench reasoning evaluator (OpenAI-style chat)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEMMA_IT = {
    "2b": "google/gemma-2-2b-it",
    "9b": "google/gemma-2-9b-it",
}

# Judge answers are short (category label / yes-no / small JSON).
_DEFAULT_MAX_NEW = 384
_DEFAULT_MAX_INPUT = 6144


def _gemma_utils():
    gemma_dir = Path(__file__).resolve().parents[1] / "server" / "gemma"
    if str(gemma_dir) not in sys.path:
        sys.path.insert(0, str(gemma_dir))
    import gemma_scope_utils as gu

    return gu


def _model_device(model: Any) -> torch.device:
    """Input device for generate(); required when model uses device_map=auto."""
    if hasattr(model, "device") and model.device is not None and model.device.type != "meta":
        return model.device
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map:
        for dev in hf_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and "cuda" in dev:
                return torch.device(dev)
    return next(model.parameters()).device


def _load_gemma_9b_judge(model_id: str) -> Any:
    """
    P100 2x16GB judge loading.
    Prefer 4bit on one GPU (GEMMA_JUDGE_9B_MODE=4bit); fp16 splits weights with headroom for KV cache.
    """
    gu = _gemma_utils()
    mode = os.environ.get(
        "GEMMA_JUDGE_9B_MODE",
        os.environ.get("GEMMA_9B_MODE", "4bit"),
    ).lower()

    if mode == "4bit":
        print("Loading Gemma 9B judge 4-bit (single GPU)...", flush=True)
        try:
            return gu._load_gemma_9b_4bit(model_id)
        except RuntimeError as exc:
            print(f"4bit failed ({exc}); falling back to fp16 split...", flush=True)
            mode = "fp16"

    n_gpu = torch.cuda.device_count()
    # ~10GiB/GPU for weights → ~6GiB left on each P100 for generate KV cache
    max_memory = {i: "10GiB" for i in range(n_gpu)}
    max_memory["cpu"] = "48GiB"
    print(
        f"Loading Gemma 9B judge fp16 (device_map=auto, max_memory={max_memory})...",
        flush=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
    )


def fold_system_for_gemma(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Gemma 2 chat templates do not support the system role; merge into user turns."""
    system_chunks: List[str] = []
    folded: List[Dict[str, str]] = []

    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_chunks.append(content)
            continue
        if role == "user" and system_chunks:
            prefix = "\n\n".join(system_chunks)
            content = f"{prefix}\n\n{content}"
            system_chunks = []
        if role in ("user", "assistant"):
            folded.append({"role": role, "content": content})
        else:
            folded.append({"role": "user", "content": content})

    if system_chunks:
        block = "\n\n".join(system_chunks)
        if folded and folded[0]["role"] == "user":
            folded[0]["content"] = f"{block}\n\n{folded[0]['content']}"
        else:
            folded.insert(0, {"role": "user", "content": block})

    return folded or [{"role": "user", "content": "Continue."}]


@dataclass
class GemmaJudgeRuntime:
    model_id: str
    gemma_size: str
    model: Any
    tokenizer: Any
    device: torch.device
    max_new_tokens: int = _DEFAULT_MAX_NEW
    max_input_tokens: int = _DEFAULT_MAX_INPUT

    def chat(self, messages: List[Dict[str, str]]) -> str:
        gemma_messages = fold_system_for_gemma(messages)
        prompt = self.tokenizer.apply_chat_template(
            gemma_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()


def load_gemma_judge(gemma_size: str) -> GemmaJudgeRuntime:
    model_id = GEMMA_IT[gemma_size]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    max_new = int(os.environ.get("GEMMA_JUDGE_MAX_NEW_TOKENS", str(_DEFAULT_MAX_NEW)))
    max_in = int(os.environ.get("GEMMA_JUDGE_MAX_INPUT_TOKENS", str(_DEFAULT_MAX_INPUT)))

    if gemma_size == "9b":
        if not torch.cuda.is_available():
            raise RuntimeError("Gemma 9B judge requires CUDA")
        model = _load_gemma_9b_judge(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        )

    device = _model_device(model)
    print(f"Gemma judge loaded: {model_id}  input_device={device}", flush=True)
    return GemmaJudgeRuntime(
        model_id=model_id,
        gemma_size=gemma_size,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new,
        max_input_tokens=max_in,
    )


def install_gemma_judge(runtime: GemmaJudgeRuntime) -> None:
    """Patch metrics.utils workflow* to call local Gemma instruct model."""
    import metrics.utils as mu

    def workflow(model_name, instruction, input_text):
        return runtime.chat(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
            ]
        )

    def workflow_multi_turn(model_name, input_text, history_messages):
        messages = list(history_messages)
        messages.append({"role": "user", "content": input_text})
        return runtime.chat(messages)

    mu.workflow = workflow  # type: ignore[assignment]
    mu.workflow_multi_turn = workflow_multi_turn  # type: ignore[assignment]
