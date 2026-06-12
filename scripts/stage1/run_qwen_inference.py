#!/usr/bin/env python3
"""
Run local Qwen3 inference on Stage-1 demo cases (P100 / transformers).

One model at a time to fit 16GB VRAM. Supports resume via incremental JSON saves.

Example (diagnosis, 8B):
  conda activate gemma_scope
  cd ~/MedRBench
  python scripts/stage1/run_qwen_inference.py \\
      --task diagnosis --model qwen3-8b

Example (treatment, 14B — run after 8B finishes):
  python scripts/stage1/run_qwen_inference.py \\
      --task treatment --model qwen3-14b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "MedRBench" / "demo_stage1_manifest.json"
INFERENCE_DIR = PROJECT_ROOT / "data" / "Stage1" / "inference"

PROMPTS = {
    "diagnosis": PROJECT_ROOT / "src" / "Inference" / "instructions" / "oracle_diagnose.txt",
    "treatment": PROJECT_ROOT / "src" / "Inference" / "instructions" / "treatment_plan_prompt.txt",
}

CASES = {
    "diagnosis": PROJECT_ROOT / "data" / "MedRBench" / "demo_diagnosis_100.json",
    "treatment": PROJECT_ROOT / "data" / "MedRBench" / "demo_treatment_100.json",
}

# Official Qwen3 post-trained checkpoints (no separate *-Instruct repo on HF).
MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
}

SYSTEM_PROMPT = "You are a professional doctor"
MAX_NEW_TOKENS = 2048


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _resolve_model_id(model_key: str, model_path: str | None) -> str:
    if model_path:
        return model_path
    return MODELS[model_key]


def _check_transformers_version() -> None:
    import transformers

    ver = transformers.__version__
    parts = tuple(int(x) for x in ver.split(".")[:2])
    if parts >= (5, 0):
        raise SystemExit(
            f"transformers {ver} requires PyTorch>=2.4; P100 env uses torch 2.1.2.\n"
            "Downgrade: pip install 'transformers>=4.51.0,<5.0' 'tokenizers>=0.21,<0.22' "
            "'huggingface_hub>=0.26,<1.0' --only-binary :all:"
        )
    if parts < (4, 51):
        raise SystemExit(
            f"Qwen3 requires transformers>=4.51.0 (you have {ver}).\n"
            "Use: conda activate qwen3_infer\n"
            "  bash scripts/stage1/setup_qwen_env.sh"
        )
    import torch

    if not torch.cuda.is_available():
        print("Warning: CUDA not available", file=sys.stderr)


def _load_model_and_tokenizer(model_id: str, model_key: str):
    _check_transformers_version()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if model_key == "qwen3-14b":
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    return model, tokenizer


def _generate(model, tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3: disable thinking chain for oracle-style answers (transformers >= 4.51).
    try:
        text = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS)
    new_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["diagnosis", "treatment"], required=True)
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local dir or HF id override (e.g. ModelScope download path)",
    )
    args = parser.parse_args()

    model_key = args.model
    model_id = _resolve_model_id(model_key, args.model_path)
    cases_path = args.cases or CASES[args.task]
    out_path = args.out or INFERENCE_DIR / f"{model_key}_{args.task}.json"

    prompt_template = PROMPTS[args.task].read_text(encoding="utf-8")
    cases = _load(cases_path)
    manifest = _load(MANIFEST)
    case_ids = manifest[args.task]["case_ids"]
    if args.limit:
        case_ids = case_ids[: args.limit]

    results: dict = {}
    if out_path.is_file() and not args.no_resume:
        results = _load(out_path)
        print(f"Resuming from {out_path} ({len(results)} cases done)")

    pending = [cid for cid in case_ids if cid in cases and cid not in results]
    if not pending:
        print("Nothing to run.")
        return

    print(f"Loading {model_id} ...")
    model, tokenizer = _load_model_and_tokenizer(model_id, model_key)

    for cid in tqdm(pending, desc=f"{model_key}/{args.task}"):
        summary = cases[cid]["generate_case"]["case_summary"]
        prompt = prompt_template.format(case=summary)
        try:
            response = _generate(model, tokenizer, prompt)
            results[cid] = {model_key: {"input": prompt, "content": response}}
        except Exception as exc:
            print(f"\nError on {cid}: {exc}", file=sys.stderr)
            results[cid] = {model_key: {"input": prompt, "error": str(exc)}}
        _save(out_path, results)

    print(f"Done. Wrote {out_path} ({len(results)} cases)")


if __name__ == "__main__":
    main()
