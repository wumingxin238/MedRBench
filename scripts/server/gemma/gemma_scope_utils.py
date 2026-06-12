"""Gemma 2 + Gemma Scope helpers for MedRBench Stage-1 pilot."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma_scope_config import DEFAULT_TOP_K, GEMMA_2B, GEMMA_9B


@dataclass
class GemmaScopeRuntime:
    model_id: str
    sae_release: str
    sae_id: str
    sae_layer: int
    tokenizer: Any
    model: Any
    sae: SAE
    device: torch.device


def _load_gemma_cfg(size: str) -> dict:
    size = size.lower()
    if size in ("2b", "2", "gemma-2-2b"):
        return GEMMA_2B
    if size in ("9b", "9", "gemma-2-9b"):
        return GEMMA_9B
    raise ValueError(f"Unknown gemma size: {size!r} (use 2b or 9b)")


def _model_device(model: Any) -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        return model.device
    return next(model.parameters()).device


def _load_gemma_9b_4bit(model_id: str) -> Any:
    """Single-GPU 4bit. Use integer device index in device_map (str cuda:0 breaks on some accelerate)."""
    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    attempts = [
        ("quantization_config, device_map=0", {
            "quantization_config": bnb,
            "device_map": {"": 0},
            "low_cpu_mem_usage": True,
        }),
        ("quantization_config, device_map=cuda:0", {
            "quantization_config": bnb,
            "device_map": "cuda:0",
            "low_cpu_mem_usage": True,
        }),
        ("quantization_config, device_map=auto", {
            "quantization_config": bnb,
            "device_map": "auto",
            "max_memory": {0: "14GiB", "cpu": "48GiB"},
            "low_cpu_mem_usage": True,
        }),
        ("load_in_4bit, device_map=0", {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "device_map": {"": 0},
            "low_cpu_mem_usage": True,
        }),
    ]
    last_exc: Optional[Exception] = None
    for label, kwargs in attempts:
        try:
            print(f"  4bit load: {label}...", flush=True)
            return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except ValueError as exc:
            last_exc = exc
            if "4-bit" not in str(exc) and "8-bit" not in str(exc):
                raise
            print(f"  4bit load failed ({label}): {exc}", flush=True)
    raise RuntimeError(f"Gemma 9B 4-bit load failed after {len(attempts)} attempts") from last_exc


def _load_gemma_9b(model_id: str) -> Any:
    """
    P100 2x16GB: default fp16 + device_map=auto (fits ~18G across 2 cards).
    Set GEMMA_9B_MODE=4bit for single-GPU 4bit (export CUDA_VISIBLE_DEVICES=0).
    """
    mode = os.environ.get("GEMMA_9B_MODE", "fp16").lower()
    if not torch.cuda.is_available():
        raise RuntimeError("Gemma 9B requires CUDA")

    if mode == "4bit":
        print("Loading Gemma 9B 4-bit (single GPU)...")
        return _load_gemma_9b_4bit(model_id)

    n_gpu = torch.cuda.device_count()
    print(f"Loading Gemma 9B fp16 (device_map=auto, {n_gpu} GPU(s))...")
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )


def load_gemma_scope(size: str = "2b", device: str = "cuda") -> GemmaScopeRuntime:
    cfg = _load_gemma_cfg(size)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])

    if size.lower() in ("9b", "9", "gemma-2-9b"):
        model = _load_gemma_9b(cfg["model_id"])
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model_dev = _model_device(model)
    sae = SAE.from_pretrained(
        release=cfg["release"],
        sae_id=cfg["sae_id"],
        device=str(model_dev),
    )
    return GemmaScopeRuntime(
        model_id=cfg["model_id"],
        sae_release=cfg["release"],
        sae_id=cfg["sae_id"],
        sae_layer=cfg["sae_layer"],
        tokenizer=tok,
        model=model,
        sae=sae,
        device=model_dev,
    )


def load_gemma_scope_2b(device: str = "cuda") -> GemmaScopeRuntime:
    return load_gemma_scope("2b", device=device)


def extract_reasoning_text(model_output: Dict[str, Any]) -> str:
    """Pull reasoning text from MedRBench oracle output fields."""
    if not isinstance(model_output, dict):
        return str(model_output)

    thinking = model_output.get("thinking_process")
    if isinstance(thinking, str) and thinking.strip():
        return thinking.strip()

    out_reasoning = model_output.get("out_reasoning")
    if isinstance(out_reasoning, str) and out_reasoning.strip():
        parts = [out_reasoning.strip()]
        out_answer = model_output.get("out_answer")
        if isinstance(out_answer, str) and out_answer.strip():
            parts.append(out_answer.strip())
        return "\n\n".join(parts)

    for key in ("content", "reasoning", "output"):
        val = model_output.get(key)
        if isinstance(val, str) and val.strip():
            return _normalize_reasoning(val.strip())
    return json.dumps(model_output, ensure_ascii=False)


def _normalize_reasoning(text: str) -> str:
    """Strip markdown code fences (common in Qwen3 outputs) before Gemma scoring."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _truncate(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars // 2] + "\n...[truncated]...\n" + text[-max_chars // 2 :]


@torch.no_grad()
def extract_sae_summary(
    rt: GemmaScopeRuntime,
    text: str,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """Forward *text* through Gemma 2B; return top SAE feature activations."""
    text = _truncate(text)
    inputs = rt.tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(rt.device) for k, v in inputs.items()}

    out = rt.model(**inputs, output_hidden_states=True)
    layer_idx = rt.sae_layer
    # hidden_states[0] is embedding; layer L is index L
    hidden = out.hidden_states[layer_idx][0, -1, :].float()
    if hidden.dim() == 1:
        hidden = hidden.unsqueeze(0)

    sae_device = next(rt.sae.parameters()).device
    acts = rt.sae.encode(hidden.to(sae_device)).squeeze()
    if acts.dim() == 0:
        acts = acts.unsqueeze(0)

    k = min(top_k, acts.numel())
    values, indices = torch.topk(acts, k=k)
    features = [
        {"feature_id": int(idx.item()), "activation": float(val.item())}
        for idx, val in zip(indices, values)
    ]
    summary_lines = [
        f"feature_{f['feature_id']}: activation={f['activation']:.4f}" for f in features
    ]
    return {
        "sae_release": rt.sae_release,
        "sae_id": rt.sae_id,
        "layer": rt.sae_layer,
        "top_features": features,
        "summary_text": "Top SAE features (layer residual stream):\n" + "\n".join(summary_lines),
    }


def build_scoring_prompt(
    case_summary: str,
    reasoning: str,
    group: str,
    sae_summary: Optional[str] = None,
) -> str:
    """Build a scoring prompt for Gemma 2 base (completion-style)."""
    case_summary = _truncate(case_summary, 2500)
    reasoning = _normalize_reasoning(_truncate(reasoning, 3000))

    header = (
        "You are a medical reasoning evaluator. "
        "Rate how good the model reasoning is for this case (1= poor, 5= excellent).\n\n"
        f"Patient case:\n{case_summary}\n\n"
        f"Model reasoning:\n{reasoning}\n\n"
    )
    if group == "sae_augmented" and sae_summary:
        header += f"SAE feature summary:\n{sae_summary}\n\n"

    # Concrete example — avoid <integer 1-5> placeholders (base model echoes them).
    header += (
        "Respond with ONE line of JSON. score must be an integer 1, 2, 3, 4, or 5.\n"
        'Example: {"score": 3, "rationale": "Structured but omits key lab findings."}\n'
        "JSON:"
    )
    return header


def _compact_rescore_suffix() -> str:
    return (
        '\nOne line only: {"score": 3, "rationale": "brief reason"}\n'
        "JSON:"
    )


@torch.no_grad()
def gemma_score(
    rt: GemmaScopeRuntime,
    prompt: str,
    max_new_tokens: int = 128,
    retries: int = 2,
) -> Dict[str, Any]:
    last_raw = ""
    for attempt in range(retries):
        use_prompt = prompt if attempt == 0 else prompt + _compact_rescore_suffix()
        inputs = rt.tokenizer(use_prompt, return_tensors="pt").to(rt.device)
        out = rt.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=rt.tokenizer.eos_token_id,
        )
        last_raw = rt.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        parsed = _parse_score_json(last_raw)
        if parsed is not None:
            return {"raw_response": last_raw.strip(), "parsed": parsed, "attempt": attempt + 1}
    return {"raw_response": last_raw.strip(), "parsed": None, "attempt": retries}


def _parse_score_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.replace("```json", "").replace("```", "").strip()
    if "<integer" in text.lower() or "<one short" in text.lower():
        text = re.sub(r"<[^>]+>", "", text)

    # Full JSON object
    for match in re.finditer(r"\{[^{}]*\"score\"[^{}]*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(match.group(0))
            if obj.get("score") is not None:
                score = int(obj["score"])
                if 1 <= score <= 5:
                    return {
                        "score": score,
                        "rationale": str(obj.get("rationale", "")),
                    }
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    patterns = [
        r'"score"\s*:\s*([1-5])',
        r"'score'\s*:\s*([1-5])",
        r"score\s*[=:]\s*([1-5])\b",
        r"Rating\s*[=:]\s*([1-5])\b",
    ]
    score = None
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            score = int(m.group(1))
            break
    if score is None:
        m = re.search(r"\b([1-5])\s*/\s*5\b", text)
        if m:
            score = int(m.group(1))
    if score is not None and 1 <= score <= 5:
        rat = re.search(r'"rationale"\s*:\s*"([^"]*)"', text)
        if not rat:
            rat = re.search(r"rationale\s*[=:]\s*\"([^\"]*)\"", text, flags=re.IGNORECASE)
        return {"score": score, "rationale": rat.group(1) if rat else ""}
    return None


def load_subject_outputs(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_pilot_cases(
    cases_path: str,
    outputs_path: str,
    subject_model: str,
    limit: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """Return list of (case_id, case_summary, reasoning_text)."""
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    outputs = load_subject_outputs(outputs_path)

    order = case_ids if case_ids is not None else list(cases.keys())
    rows: List[Tuple[str, str, str]] = []
    for case_id in order:
        if case_id not in cases or case_id not in outputs:
            continue
        case_entry = cases[case_id]
        out_entry = outputs[case_id]
        if subject_model not in out_entry:
            continue
        summary = case_entry.get("generate_case", {}).get("case_summary", "")
        reasoning = extract_reasoning_text(out_entry[subject_model])
        rows.append((case_id, summary, reasoning))
        if limit and len(rows) >= limit:
            break
    return rows
