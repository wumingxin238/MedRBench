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
import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def _apply_early_gpu_from_argv() -> None:
    """Set CUDA_VISIBLE_DEVICES before importing torch."""
    if len(sys.argv) >= 2 and sys.argv[1] == "--probe-load":
        return
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu-id" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return


_apply_early_gpu_from_argv()

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "MedRBench" / "demo_stage1_manifest.json"
STAGE1_DIR = PROJECT_ROOT / "data" / "Stage1"
INFERENCE_DIR = STAGE1_DIR / "inference"

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
# Pre-quantized AWQ (~9GB VRAM); avoids bitsandbytes (problematic on P100 + bnb 0.42).
AWQ_MODELS = {
    "qwen3-14b": "Qwen/Qwen3-14B-AWQ",
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


def _print_gpu_mem(prefix: str = "") -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(
            f"{prefix}GPU cuda:{i}  free={free / 1024**3:.2f}GiB  total={total / 1024**3:.2f}GiB",
            file=sys.stderr,
        )


def _reset_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _try_load(model_id: str, label: str, kwargs: dict):
    print(f"  try {label} ...", file=sys.stderr)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        print(f"  loaded with {label}", file=sys.stderr)
        return model
    except Exception:
        _reset_cuda()
        raise


def _offload_dir() -> Path:
    path = STAGE1_DIR / ".qwen14b_offload"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_14b_strategies(quant_mode: str) -> dict[str, dict]:
    from transformers import BitsAndBytesConfig

    n_gpu = torch.cuda.device_count()
    offload = str(_offload_dir())

    bnb4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    bnb8_offload = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    strategies: dict[str, dict] = {}

    if quant_mode in ("auto", "4bit"):
        strategies.update(
            {
                "4bit-offload-disk": {
                    "quantization_config": bnb4,
                    "device_map": "auto",
                    "max_memory": {0: "12GiB", "cpu": "48GiB"},
                    "offload_folder": offload,
                    "offload_state_dict": True,
                },
                "4bit-auto-cpu-offload": {
                    "quantization_config": bnb4,
                    "device_map": "auto",
                    "max_memory": {0: "12GiB", "cpu": "48GiB"},
                },
                "4bit-cuda0-str": {
                    "quantization_config": bnb4,
                    "device_map": "cuda:0",
                },
            }
        )

    if quant_mode in ("auto", "8bit"):
        strategies.update(
            {
                "8bit-auto-cpu-offload": {
                    "quantization_config": bnb8_offload,
                    "device_map": "auto",
                    "max_memory": {0: "12GiB", "cpu": "48GiB"},
                    "offload_folder": offload,
                    "offload_state_dict": True,
                },
            }
        )

    # fp16 needs both GPUs mostly free (~28GB). Do not use while Gemma holds GPU0.
    if quant_mode in ("fp16-split",) and n_gpu >= 2:
        strategies["fp16-2gpu-balanced"] = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_memory": {i: "15GiB" for i in range(n_gpu)},
        }

    if quant_mode == "fp16-asymmetric" and n_gpu >= 2:
        strategies["fp16-gpu0-limited-gpu1-main"] = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "max_memory": {0: "0GiB", 1: "15GiB", "cpu": "48GiB"},
        }

    return strategies


def _probe_load_subprocess(model_id: str, strategy_name: str) -> tuple[bool, str]:
    """Try one load strategy in a fresh process so failed loads release GPU memory."""
    script = str(Path(__file__).resolve())
    cmd = [sys.executable, script, "--probe-load", strategy_name, model_id]
    env = os.environ.copy()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    detail = (result.stderr or result.stdout or "").strip()
    if len(detail) > 300:
        detail = detail[:300] + "..."
    return result.returncode == 0, detail


def _probe_load_main() -> None:
    strategy_name = sys.argv[2]
    model_id = sys.argv[3]
    quant_mode = os.environ.get("QWEN14B_PROBE_QUANT", "auto")
    strategies = _build_14b_strategies(quant_mode)
    if strategy_name not in strategies:
        raise SystemExit(f"Unknown strategy: {strategy_name}")
    _check_runtime_deps(need_quant=True)
    model = _try_load(model_id, strategy_name, strategies[strategy_name])
    del model
    _reset_cuda()


def _cuda_preflight() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    try:
        _print_gpu_mem("preflight ")
        x = torch.zeros(1, device="cuda:0")
        del x
        torch.cuda.synchronize()
        print("preflight cuda:0 ok", file=sys.stderr)
    except RuntimeError as exc:
        raise SystemExit(
            f"CUDA preflight failed on visible cuda:0: {exc}\n"
            "Try a fresh SSH/tmux session with: export CUDA_VISIBLE_DEVICES=1\n"
            "Quick test:\n"
            "  CUDA_VISIBLE_DEVICES=1 python -c \"import torch; torch.zeros(1).cuda()\"\n"
            "If that fails while nvidia-smi shows GPU1 free, ask admin for: nvidia-smi --gpu-reset -i 1\n"
            "Recommended: --quant-mode awq (no bitsandbytes)"
        ) from exc


def _load_14b_awq(awq_id: str):
    _cuda_preflight()
    print(f"Loading AWQ (no bitsandbytes): {awq_id}", file=sys.stderr)
    return AutoModelForCausalLM.from_pretrained(
        awq_id,
        device_map="auto",
        trust_remote_code=True,
    )


def _load_14b_bnb(model_id: str, quant_mode: str):
    n_gpu = torch.cuda.device_count()
    print(f"Loading 14B on {n_gpu} visible GPU(s), quant_mode={quant_mode}", file=sys.stderr)
    _print_gpu_mem("mem ")
    os.environ["QWEN14B_PROBE_QUANT"] = quant_mode

    strategies = _build_14b_strategies(quant_mode)
    if not strategies:
        raise SystemExit(f"No load strategy for quant_mode={quant_mode!r} with {n_gpu} GPU(s)")

    winning: str | None = None
    last_detail = ""
    for name in strategies:
        ok, detail = _probe_load_subprocess(model_id, name)
        if detail:
            print(f"  probe {name}: {detail}", file=sys.stderr)
        if ok:
            winning = name
            break
        last_detail = detail
        _reset_cuda()

    if winning is None:
        raise RuntimeError(
            "All 14B load strategies failed in isolated probes.\n"
            f"Last error: {last_detail}\n\n"
            "Recommended: --quant-mode awq (pip install autoawq)\n"
            "Or wait for Gemma to finish, then: --quant-mode fp16-split (both GPUs free)."
        )

    _print_gpu_mem("mem before final load ")
    return _try_load(model_id, winning, strategies[winning])


def _load_14b(model_id: str, quant_mode: str, model_path: str | None):
    awq_id = model_path or AWQ_MODELS.get("qwen3-14b", model_id)

    if quant_mode in ("auto", "awq"):
        try:
            return _load_14b_awq(awq_id)
        except Exception as exc:
            if quant_mode == "awq":
                raise RuntimeError(
                    f"AWQ load failed for {awq_id}: {exc}\n"
                    "Install: pip install autoawq\n"
                    "Or use base model: --quant-mode 4bit (needs bitsandbytes>=0.43)"
                ) from exc
            print(f"AWQ load failed ({exc}); trying bitsandbytes...", file=sys.stderr)
            _reset_cuda()

    if quant_mode in ("auto", "4bit", "8bit", "fp16-split", "fp16-asymmetric"):
        return _load_14b_bnb(model_id, quant_mode)

    raise SystemExit(f"Unknown quant_mode for 14B: {quant_mode}")


def _load_model_and_tokenizer(
    model_id: str, model_key: str, quant_mode: str = "auto", model_path: str | None = None
):
    need_quant = model_key == "qwen3-14b"
    _check_runtime_deps(need_quant=need_quant and quant_mode not in ("awq", "auto"))
    tok_id = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
    if need_quant:
        model = _load_14b(model_id, quant_mode, model_path)
    else:
        load_id = model_path or model_id
        model = AutoModelForCausalLM.from_pretrained(
            load_id,
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
        "--gpu-id",
        type=str,
        default=None,
        help="Physical GPU index (sets CUDA_VISIBLE_DEVICES before torch init), e.g. 1",
    )
    parser.add_argument(
        "--quant-mode",
        choices=["auto", "awq", "4bit", "8bit", "fp16-split", "fp16-asymmetric"],
        default="auto",
        help=(
            "14B only. auto=AWQ first (recommended on P100), then bnb fallbacks. "
            "awq uses Qwen/Qwen3-14B-AWQ (~9GB, no bitsandbytes). "
            "fp16-split needs both GPUs free (stop Gemma first)."
        ),
    )
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

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
    model, tokenizer = _load_model_and_tokenizer(
        model_id, model_key, args.quant_mode, args.model_path
    )

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
    if len(sys.argv) >= 2 and sys.argv[1] == "--probe-load":
        try:
            _probe_load_main()
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    main()
