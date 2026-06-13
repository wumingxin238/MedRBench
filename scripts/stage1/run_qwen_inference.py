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


def _check_runtime_deps(*, need_quant: bool = False) -> None:
    import numpy as np
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

    np_major = int(str(np.__version__).split(".")[0])
    if np_major >= 2:
        raise SystemExit(
            f"NumPy {np.__version__} breaks torch 2.1 wheels on this env.\n"
            "Fix: pip install 'numpy==1.26.4' --only-binary :all:\n"
            "  or: conda install -y numpy=1.26.4"
        )

    if need_quant:
        import bitsandbytes as bnb

        print(f"bitsandbytes {bnb.__version__}", file=sys.stderr)

    if not torch.cuda.is_available():
        print("Warning: CUDA not available", file=sys.stderr)


def _model_input_device(model) -> torch.device:
    try:
        dev = model.device
        if dev.type != "meta":
            return dev
    except Exception:
        pass
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda:0")


def _try_load(model_id: str, label: str, kwargs: dict):
    print(f"  try {label} ...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **kwargs,
    )
    print(f"  loaded with {label}", file=sys.stderr)
    return model


def _fourbit_attempts(bnb_config) -> list[tuple[str, dict]]:
    """Same fallbacks as gemma_scope_utils._load_gemma_9b_4bit (bnb 0.42 on P100)."""
    return [
        (
            "4bit-auto-cpu-offload",
            {
                "quantization_config": bnb_config,
                "device_map": "auto",
                "max_memory": {0: "14GiB", "cpu": "48GiB"},
            },
        ),
        (
            "4bit-cuda0-str",
            {
                "quantization_config": bnb_config,
                "device_map": "cuda:0",
            },
        ),
        (
            "4bit-device0-dict",
            {
                "quantization_config": bnb_config,
                "device_map": {"": 0},
            },
        ),
        (
            "4bit-legacy-kwargs",
            {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "device_map": "auto",
                "max_memory": {0: "14GiB", "cpu": "48GiB"},
            },
        ),
    ]


def _load_14b_model(model_id: str, quant_mode: str):
    from transformers import BitsAndBytesConfig

    n_gpu = torch.cuda.device_count()
    print(f"Loading 14B on {n_gpu} visible GPU(s), quant_mode={quant_mode}", file=sys.stderr)

    bnb4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    bnb8 = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    strategies: list[tuple[str, dict]] = []

    if quant_mode in ("auto", "4bit"):
        strategies.extend(_fourbit_attempts(bnb4))

    if quant_mode in ("auto", "8bit"):
        strategies.extend(
            [
                (
                    "8bit-auto-cpu-offload",
                    {
                        "quantization_config": bnb8,
                        "device_map": "auto",
                        "max_memory": {0: "14GiB", "cpu": "48GiB"},
                    },
                ),
                (
                    "8bit-gpu-only",
                    {
                        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                        "device_map": "auto",
                        "max_memory": {0: "15GiB"},
                    },
                ),
            ]
        )

    if quant_mode in ("auto", "fp16-asymmetric") and n_gpu >= 2:
        # GPU0 may be busy (e.g. Gemma 9B ~11GB); put most weights on GPU1.
        strategies.append(
            (
                "fp16-gpu0-limited-gpu1-main",
                {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "max_memory": {0: "4GiB", 1: "15GiB", "cpu": "48GiB"},
                },
            )
        )

    if quant_mode in ("auto", "fp16-split") and n_gpu >= 2:
        strategies.append(
            (
                "fp16-2gpu-balanced",
                {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "max_memory": {i: "15GiB" for i in range(n_gpu)},
                },
            )
        )

    if not strategies:
        raise SystemExit(f"No load strategy for quant_mode={quant_mode!r} with {n_gpu} GPU(s)")

    last_err: Exception | None = None
    for name, kwargs in strategies:
        try:
            return _try_load(model_id, name, kwargs)
        except (ValueError, RuntimeError, OSError) as exc:
            last_err = exc
            print(f"  {name} failed: {exc}", file=sys.stderr)
    assert last_err is not None
    raise last_err


def _load_model_and_tokenizer(model_id: str, model_key: str, quant_mode: str = "auto"):
    need_quant = model_key == "qwen3-14b"
    _check_runtime_deps(need_quant=need_quant)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if need_quant:
        model = _load_14b_model(model_id, quant_mode)
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
    model_inputs = tokenizer([text], return_tensors="pt").to(_model_input_device(model))
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
    parser.add_argument(
        "--quant-mode",
        choices=["auto", "4bit", "8bit", "fp16-split", "fp16-asymmetric"],
        default="auto",
        help=(
            "14B only. auto=4bit fallbacks→8bit→fp16-asymmetric→fp16-split. "
            "Use fp16-asymmetric when GPU0 runs Gemma (unset CUDA_VISIBLE_DEVICES)."
        ),
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
    model, tokenizer = _load_model_and_tokenizer(model_id, model_key, args.quant_mode)

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
