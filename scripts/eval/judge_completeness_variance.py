#!/usr/bin/env python3
"""
Re-run completeness hit-check on selected cases to measure judge variance.

Requires local judge (EVAL_BACKEND=vllm, server on EVAL_BASE_URL).
Default: temperature=0 (greedy, should be deterministic on GPU).
Optional: --temperature >0 with --seed for sampling stability tests.

Usage:
  source scripts/server/config/eval_config.env   # or set EVAL_* on Windows
  python scripts/eval/judge_completeness_variance.py --cases PMC11625232 PMC11321471 --repeats 5
  python scripts/eval/judge_completeness_variance.py --from-analysis top_qwq --n 8 --repeats 10
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "src/Evaluation"
# metrics.* imports expect cwd/PYTHONPATH = src/Evaluation (same as oracle_diagnose_reasoning.py)
sys.path.insert(0, str(EV))

from metrics.reasoning_eval import (  # noqa: E402
    check_step_hit,
    eval_reasoning_completeness,
)


def _combined_reasoning(result_json: dict) -> str:
    content = (result_json.get("results") or {}).get("content") or ""
    if "### Answer:" in content:
        content = content.split("### Answer:")[0]
    if "### Resoning:" in content:
        content = content.split("### Resoning:")[-1]
    elif "### Reasoning:" in content:
        content = content.split("### Reasoning:")[-1]
    return content.strip()


def gt_reasoning_text(case_json: dict) -> str:
    gen = case_json.get("generate_case") or {}
    dd = gen.get("differential_diagnosis") or ""
    fd = gen.get("final_diagnosis") or ""
    return f"{dd}\n Final diagnosis:\n{fd}"


def load_case(model: str, cid: str) -> dict:
    path = EV / f"reasoning_results_qwen_judge_paper_957/{model}/{cid}.json"
    return json.load(open(path, encoding="utf-8"))


def pick_cases_from_analysis(kind: str, n: int) -> list[str]:
    csv_path = ROOT / "docs/artifacts/completeness_gap_by_case.csv"
    if not csv_path.exists():
        raise SystemExit("Run analyze_completeness_gap_by_case.py first.")
    import csv

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if kind == "top_qwq":
        rows.sort(key=lambda r: float(r["delta"]), reverse=True)
    elif kind == "top_gemini":
        rows.sort(key=lambda r: float(r["delta"]))
    elif kind == "gemini_zero":
        rows = [r for r in rows if float(r["gemini_recall"]) < 0.01]
        rows.sort(key=lambda r: float(r["qwq_recall"]), reverse=True)
    else:
        raise ValueError(kind)
    return [r["id"] for r in rows[:n]]


def run_completeness_once(gt_text: str, pred_text: str, model: str) -> tuple[float, list[bool]]:
    out = eval_reasoning_completeness(gt_text, pred_text, evaluation_model=model)
    hits = [x["hit"] for x in out["ground_truth_steps"]]
    return out["recall_score"], hits


def run_hit_only(gt_steps: list[str], pred_text: str, model: str) -> list[bool]:
    return [
        check_step_hit(step, pred_text, evaluation_model=model)
        for step in gt_steps
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", help="PMC IDs")
    parser.add_argument("--from-analysis", choices=["top_qwq", "top_gemini", "gemini_zero"])
    parser.add_argument("-n", type=int, default=5, help="Cases when using --from-analysis")
    parser.add_argument("--models", nargs="+", default=["gemini2-ft", "qwq"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["full", "hit_only"], default="hit_only",
                        help="full=re-split GT each run; hit_only=fixed GT steps from saved JSON")
    parser.add_argument("--eval-model", default=None)
    args = parser.parse_args()

    if args.temperature is not None:
        os.environ["EVAL_TEMPERATURE"] = str(args.temperature)
    else:
        os.environ.setdefault("EVAL_TEMPERATURE", "0")

    if args.cases:
        case_ids = args.cases
    elif args.from_analysis:
        case_ids = pick_cases_from_analysis(args.from_analysis, args.n)
    else:
        case_ids = pick_cases_from_analysis("top_qwq", 3)

    eval_model = args.eval_model or os.environ.get("EVAL_MODEL", "gpt-4o-2024-11-20")
    backend = os.environ.get("EVAL_BACKEND", "openai")
    base_url = os.environ.get("EVAL_BASE_URL", "")

    print("=" * 72)
    print(f"Judge variance probe  repeats={args.repeats}  temp={os.environ['EVAL_TEMPERATURE']}")
    print(f"EVAL_BACKEND={backend}  BASE_URL={base_url}  MODEL={eval_model}")
    print(f"Cases: {case_ids}")
    print("=" * 72)

    # Optional: set torch seed when sampling (local server may ignore)
    if float(os.environ["EVAL_TEMPERATURE"]) > 0:
        try:
            import torch

            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            print(f"torch seed={args.seed}")
        except ImportError:
            pass

    summary_rows = []

    for cid in case_ids:
        print(f"\n--- {cid} ---")
        for model in args.models:
            try:
                data = load_case(model, cid)
            except FileNotFoundError:
                print(f"  [{model}] missing, skip")
                continue

            saved_recall = float(data.get("recall") or 0)
            gt_steps_saved = [x["step"] for x in (data.get("gt_reasoning_eval") or [])]
            pred_text = _combined_reasoning(data)
            gt_text = gt_reasoning_text(data)

            recalls, hit_vectors = [], []
            t0 = time.time()
            for rep in range(args.repeats):
                if args.mode == "full":
                    recall, hits = run_completeness_once(gt_text, pred_text, eval_model)
                else:
                    hits = run_hit_only(gt_steps_saved, pred_text, eval_model)
                    recall = sum(hits) / len(hits) if hits else 0.0
                recalls.append(recall)
                hit_vectors.append(tuple(hits))
                print(f"  [{model}] rep {rep+1}/{args.repeats} recall={100*recall:.1f}% hits={sum(hits)}/{len(hits)}")

            elapsed = time.time() - t0
            unique_vectors = len(set(hit_vectors))
            unique_recalls = len(set(round(r, 6) for r in recalls))

            row = {
                "id": cid,
                "model": model,
                "saved_recall": saved_recall,
                "mean_recall": statistics.mean(recalls),
                "stdev_recall": statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
                "min_recall": min(recalls),
                "max_recall": max(recalls),
                "unique_hit_patterns": unique_vectors,
                "unique_recall_values": unique_recalls,
                "repeats": args.repeats,
                "elapsed_s": round(elapsed, 1),
            }
            summary_rows.append(row)

            stable = "STABLE" if unique_vectors == 1 else f"UNSTABLE ({unique_vectors} patterns)"
            match_saved = "matches saved" if abs(row["mean_recall"] - saved_recall) < 0.02 else "differs from saved"
            print(
                f"  [{model}] {stable}  mean={100*row['mean_recall']:.2f}% "
                f"std={100*row['stdev_recall']:.2f}pp  range=[{100*row['min_recall']:.1f},{100*row['max_recall']:.1f}]%  "
                f"{match_saved} (saved {100*saved_recall:.1f}%)"
            )

    out_path = ROOT / "docs/artifacts/judge_completeness_variance.json"
    out_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
