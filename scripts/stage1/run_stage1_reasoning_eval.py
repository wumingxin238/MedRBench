#!/usr/bin/env python3
"""
Stage-1 reasoning evaluation using MedRBench metrics (Efficiency / Factuality / Completeness).

Two scoring groups (2.2):
  direct              — evaluate reasoning steps only
  inference_augmented — evaluate reasoning + subject model final inference (appended as last step)

Judge: Gemma 2B/9B instruct (local in-process) or any OpenAI-compatible server via EVAL_* env.

Examples:
  # Gemma 2B local judge (no openai package needed)
  conda activate gemma_scope
  export EVAL_DISABLE_WEB_SEARCH=1
  python scripts/stage1/run_stage1_reasoning_eval.py \\
      --task diagnosis --subject-model o3-mini --judge gemma-local --gemma-size 2b

  # Gemma 9B judge on P100
  #   4bit (recommended): export CUDA_VISIBLE_DEVICES=0 GEMMA_JUDGE_9B_MODE=4bit
  #   fp16 dual-GPU fallback: unset CUDA_VISIBLE_DEVICES, GEMMA_JUDGE_9B_MODE=fp16
  export CUDA_VISIBLE_DEVICES=0
  export GEMMA_JUDGE_9B_MODE=4bit
  export EVAL_DISABLE_WEB_SEARCH=1
  python scripts/stage1/run_stage1_reasoning_eval.py \\
      --task diagnosis --subject-model qwen3-8b --judge gemma-local --gemma-size 9b --limit 1
  bash scripts/stage1/setup_gemma_reasoning_eval.sh
  export EVAL_BACKEND=vllm
  export EVAL_BASE_URL=http://127.0.0.1:8000/v1
  export EVAL_MODEL=Qwen/Qwen2.5-7B-Instruct
  python scripts/stage1/run_stage1_reasoning_eval.py \\
      --task diagnosis --subject-model qwen3-8b --judge server --group direct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
EVAL_DIR = PROJECT_ROOT / "src" / "Evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from stage1_subject_text import (  # noqa: E402
    build_reasoning_steps,
    gt_answer_for_case,
    gt_reasoning_for_case,
)

MANIFEST = PROJECT_ROOT / "data" / "MedRBench" / "demo_stage1_manifest.json"
STAGE1_DIR = PROJECT_ROOT / "data" / "Stage1"
DEFAULT_OUT_DIR = STAGE1_DIR / "reasoning_eval"

CASES = {
    "diagnosis": PROJECT_ROOT / "data" / "MedRBench" / "demo_diagnosis_100.json",
    "treatment": PROJECT_ROOT / "data" / "MedRBench" / "demo_treatment_100.json",
}

SUBJECTS = {
    "diagnosis": STAGE1_DIR / "oracle_diagnosis_subjects.json",
    "treatment": STAGE1_DIR / "oracle_treatment_subjects.json",
}

GROUPS = ("direct", "inference_augmented")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _out_path(out_dir: Path, task: str, judge_tag: str, subject: str, group: str) -> Path:
    subj = subject.replace("/", "_")
    return out_dir / f"{task}_{judge_tag}_{subj}_{group}.json"


def _setup_judge(args) -> str:
    if args.judge == "gemma-local":
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        from gemma_judge_backend import install_gemma_judge, load_gemma_judge

        rt = load_gemma_judge(args.gemma_size)
        install_gemma_judge(rt)
        os.environ["EVAL_BACKEND"] = "gemma_local"
        return f"gemma-{args.gemma_size}-it"
    return os.environ.get("EVAL_MODEL", "gpt-4o-2024-11-20")


def evaluate_one_case(
    case_id: str,
    case: dict,
    result: dict,
    *,
    task: str,
    subject_model: str,
    group: str,
    evaluation_model: str,
    use_web_search: bool,
) -> dict:
    from metrics.reasoning_eval import (
        eval_reasoning_completeness,
        eval_reasoning_efficiency_factuality,
    )

    case_info = case["generate_case"]["case_summary"]
    gt_answer = gt_answer_for_case(case, task)
    gt_reasoning = gt_reasoning_for_case(case, task)
    is_treatment = task == "treatment"

    steps, combined = build_reasoning_steps(result, subject_model, group=group)
    if not steps:
        return {
            "status": "skipped",
            "error": "no_reasoning_steps",
            "efficiency": None,
            "factulity": None,
            "recall": None,
        }

    ef = eval_reasoning_efficiency_factuality(
        case_info=case_info,
        pred_reasoning_steps_list=steps,
        gt_answer=gt_answer,
        is_treatment=is_treatment,
        evaluation_model=evaluation_model,
        use_web_search=use_web_search,
    )
    comp = eval_reasoning_completeness(
        gt_reasoning=gt_reasoning,
        pred_reasoning_steps_string=combined,
        evaluation_model=evaluation_model,
    )

    return {
        "status": "ok",
        "group": group,
        "step_count": len(steps),
        "efficiency": ef["efficiency_score"],
        "factulity": ef["factuality_score"],
        "recall": comp["recall_score"],
        "reasoning_eval": ef["evaluated_steps"],
        "gt_reasoning_eval": comp["ground_truth_steps"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["diagnosis", "treatment"], required=True)
    parser.add_argument("--subject-model", required=True)
    parser.add_argument(
        "--group",
        choices=list(GROUPS) + ["both"],
        default="both",
        help="Scoring group (default: run direct + inference_augmented)",
    )
    parser.add_argument(
        "--judge",
        choices=["gemma-local", "server"],
        default="gemma-local",
        help="gemma-local loads Gemma-it in-process; server uses EVAL_* env",
    )
    parser.add_argument("--gemma-size", choices=["2b", "9b"], default="2b")
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--outputs", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Max cases (0 = all)")
    parser.add_argument("--no-web-search", action="store_true", default=True)
    parser.add_argument("--web-search", action="store_true", help="Enable Bing search for factuality")
    args = parser.parse_args()

    if args.web_search:
        args.no_web_search = False
    if args.no_web_search:
        os.environ["EVAL_DISABLE_WEB_SEARCH"] = "1"

    from metrics.eval_flags import web_search_enabled

    use_web_search = web_search_enabled()
    judge_tag = _setup_judge(args)
    evaluation_model = judge_tag if args.judge == "gemma-local" else os.environ.get("EVAL_MODEL", judge_tag)

    cases_path = args.cases or CASES[args.task]
    outputs_path = args.outputs or SUBJECTS[args.task]
    cases = _load(cases_path)
    outputs = _load(outputs_path)

    groups = list(GROUPS) if args.group == "both" else [args.group]

    for group in groups:
        out_path = _out_path(args.out_dir, args.task, judge_tag.replace("/", "_"), args.subject_model, group)
        doc = _load(out_path) if out_path.is_file() else {}
        doc.setdefault("meta", {})
        doc.setdefault("cases", {})
        meta = doc["meta"]
        meta.update(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "task": args.task,
                "subject_model": args.subject_model,
                "group": group,
                "judge": args.judge,
                "judge_model": evaluation_model,
                "gemma_size": args.gemma_size if args.judge == "gemma-local" else None,
                "metrics": ["efficiency", "factulity", "recall"],
                "web_search": use_web_search,
            }
        )

        case_ids = list(cases.keys())
        if args.limit > 0:
            case_ids = case_ids[: args.limit]

        err_count = sum(1 for c in doc["cases"].values() if c.get("status") == "error")
        if err_count:
            print(f"  ({err_count} prior errors will be retried on re-run)")

        done = 0
        for i, cid in enumerate(case_ids, 1):
            if doc["cases"].get(cid, {}).get("status") == "ok":
                continue
            if cid not in outputs or args.subject_model not in outputs[cid]:
                doc["cases"][cid] = {"status": "skipped", "error": "missing_subject_output"}
                continue

            print(f"[{group}] {i}/{len(case_ids)} {cid}", flush=True)
            try:
                row = evaluate_one_case(
                    cid,
                    cases[cid],
                    outputs[cid][args.subject_model],
                    task=args.task,
                    subject_model=args.subject_model,
                    group=group,
                    evaluation_model=evaluation_model,
                    use_web_search=use_web_search,
                )
                doc["cases"][cid] = row
                if row.get("status") == "ok":
                    done += 1
            except Exception as exc:
                doc["cases"][cid] = {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            _save(out_path, doc)

        ok_rows = [c for c in doc["cases"].values() if c.get("status") == "ok"]
        if ok_rows:
            meta["mean_efficiency"] = sum(r["efficiency"] for r in ok_rows) / len(ok_rows)
            meta["mean_factulity"] = sum(r["factulity"] for r in ok_rows) / len(ok_rows)
            meta["mean_recall"] = sum(r["recall"] for r in ok_rows) / len(ok_rows)
        meta["completed_ok"] = len(ok_rows)
        meta["total_cases"] = len(case_ids)
        _save(out_path, doc)
        print(f"Wrote {out_path} ({len(ok_rows)} ok / {len(case_ids)} cases)")
        if len(ok_rows) == 0 and doc["cases"]:
            sample = next(c for c in doc["cases"].values() if c.get("error"))
            if sample:
                print(f"  All failed. Example error: {sample.get('error')}")


if __name__ == "__main__":
    main()
