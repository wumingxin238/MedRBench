#!/usr/bin/env python3
"""Stage-2 diagnosis accuracy with local Gemma-9B judge (in-process)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "src" / "Evaluation"
STAGE2 = PROJECT_ROOT / "data" / "Stage2"
STAGE1 = PROJECT_ROOT / "scripts" / "stage1"

DEFAULT_CASES = PROJECT_ROOT / "data" / "MedRBench" / "diagnosis_400.json"
DEFAULT_OUTPUTS = STAGE2 / "oracle_diagnosis_subjects.json"
DEFAULT_OUT = STAGE2 / "acc_results_gemma"
DEFAULT_MODEL = "qwen3-14b-thinking"


def _extract_answer(text: str) -> str:
    if "### Answer" in text:
        return text.split("### Answer")[-1].replace("\n", "").replace(":", "").strip()
    return text.strip()


def _is_valid_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if row.get("acc_error"):
        return False
    return row.get("acc_judge", "").startswith("gemma-local-")


def _setup_gemma_eval_accuracy(gemma_size: str):
    """Patch metrics.utils.workflow AND outcome_accuracy_eval.workflow."""
    sys.path.insert(0, str(EVAL_DIR))
    sys.path.insert(0, str(STAGE1))
    import metrics.outcome_accuracy_eval as oae  # noqa: E402
    import metrics.utils as mu  # noqa: E402
    from gemma_judge_backend import install_gemma_judge, load_gemma_judge  # noqa: E402

    runtime = load_gemma_judge(gemma_size)
    install_gemma_judge(runtime)
    oae.workflow = mu.workflow
    os.environ["EVAL_BACKEND"] = "gemma_local"
    return oae.eval_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-model", default=DEFAULT_MODEL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gemma-size", default="9b", choices=("2b", "9b"))
    args = parser.parse_args()

    os.environ.setdefault("EVAL_DISABLE_WEB_SEARCH", "1")
    os.environ.setdefault("GEMMA_JUDGE_9B_MODE", os.environ.get("GEMMA_JUDGE_9B_MODE", "fp16"))

    eval_accuracy = _setup_gemma_eval_accuracy(args.gemma_size)

    from eval_io import load_model_outputs  # noqa: E402

    out_dir = args.out_dir / args.subject_model
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    model_outputs = load_model_outputs(str(args.outputs.resolve()), args.subject_model, embedded=False)

    valid = sum(1 for p in out_dir.glob("*.json") if _is_valid_result(p))
    todo = [
        cid
        for cid in patient_cases
        if not _is_valid_result(out_dir / f"{cid}.json")
        and cid in model_outputs
        and args.subject_model in model_outputs[cid]
    ]

    print(
        f"==> {args.subject_model} accuracy  judge=gemma-local-{args.gemma_size}  "
        f"({valid}/{len(patient_cases)} ok, {len(todo)} todo)",
        flush=True,
    )
    if not todo:
        return

    ok_n = err_n = 0
    for i, cid in enumerate(todo, 1):
        case = dict(patient_cases[cid])
        case["id"] = cid
        case["results"] = model_outputs[cid][args.subject_model]
        pred = _extract_answer(case["results"].get("content", "") or "")
        gt = case["generate_case"]["diagnosis_results"]
        out_path = out_dir / f"{cid}.json"
        try:
            ok = eval_accuracy(pred, gt, evaluation_model=f"gemma-{args.gemma_size}-it")
            case["accuracy"] = bool(ok)
            case["acc_judge"] = f"gemma-local-{args.gemma_size}"
            case.pop("acc_error", None)
            out_path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
            ok_n += 1
        except Exception as exc:
            err_n += 1
            print(f"  ERROR {cid}: {exc}", flush=True)
            if out_path.is_file():
                out_path.unlink()
        if i % 10 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] last={cid} saved={ok_n} err={err_n}", flush=True)

    print(f"==> done {args.subject_model} -> {out_dir}  saved={ok_n} err={err_n}", flush=True)


if __name__ == "__main__":
    main()
