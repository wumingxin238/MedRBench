#!/usr/bin/env python3
"""
Stage-1 Gemma Scope batch evaluation (direct vs sae_augmented).

Scores subject model reasoning for all demo cases. Supports resume.

Example (diagnosis, Gemma 2B, one subject model):
  conda activate gemma_scope
  cd ~/MedRBench
  python scripts/stage1/prepare_subject_outputs.py --task diagnosis
  python scripts/stage1/run_gemma_scope_eval.py \\
      --task diagnosis --gemma-size 2b --subject-model deepseek-r1

Run all four subject models sequentially:
  for m in deepseek-r1 o3-mini qwen3-8b qwen3-14b; do
    python scripts/stage1/run_gemma_scope_eval.py \\
        --task diagnosis --gemma-size 2b --subject-model $m
  done
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GEMMA_DIR = SCRIPT_DIR.parent / "server" / "gemma"
if str(GEMMA_DIR) not in sys.path:
    sys.path.insert(0, str(GEMMA_DIR))

from gemma_scope_utils import (  # noqa: E402
    build_scoring_prompt,
    extract_sae_summary,
    gemma_score,
    iter_pilot_cases,
    load_gemma_scope,
)

PROJECT_ROOT = SCRIPT_DIR.parents[1]
MANIFEST = PROJECT_ROOT / "data" / "MedRBench" / "demo_stage1_manifest.json"
STAGE1_DIR = PROJECT_ROOT / "data" / "Stage1"
DEFAULT_OUT_DIR = STAGE1_DIR / "gemma_scope"

CASES = {
    "diagnosis": PROJECT_ROOT / "data" / "MedRBench" / "demo_diagnosis_100.json",
    "treatment": PROJECT_ROOT / "data" / "MedRBench" / "demo_treatment_100.json",
}

SUBJECTS = {
    "diagnosis": STAGE1_DIR / "oracle_diagnosis_subjects.json",
    "treatment": STAGE1_DIR / "oracle_treatment_subjects.json",
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _out_path(out_dir: Path, task: str, gemma_size: str, subject: str) -> Path:
    return out_dir / f"{task}_{gemma_size}_{subject.replace('/', '_')}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["diagnosis", "treatment"], required=True)
    parser.add_argument("--gemma-size", choices=["2b", "9b"], default="2b")
    parser.add_argument("--subject-model", required=True)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--outputs", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["direct", "sae_augmented"],
        choices=["direct", "sae_augmented"],
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--rescore-failed",
        action="store_true",
        help="Re-run scoring only for cases/groups where JSON parse failed",
    )
    args = parser.parse_args()

    cases_path = args.cases or CASES[args.task]
    outputs_path = args.outputs or SUBJECTS[args.task]
    out_path = _out_path(args.out_dir, args.task, args.gemma_size, args.subject_model)

    if not outputs_path.is_file():
        raise SystemExit(
            f"Subject outputs not found: {outputs_path}\n"
            f"Run: python scripts/stage1/prepare_subject_outputs.py --task {args.task}"
        )

    manifest = _load(MANIFEST)
    case_ids = manifest[args.task]["case_ids"]

    rows = iter_pilot_cases(
        str(cases_path),
        str(outputs_path),
        args.subject_model,
        limit=args.limit,
        case_ids=case_ids,
    )

    if args.no_resume:
        results = {"meta": {}, "cases": {}}
    elif args.rescore_failed and out_path.is_file():
        results = _load(out_path)
        print(f"Rescore-failed mode: {out_path}")
    elif out_path.is_file():
        results = _load(out_path)
        print(f"Resuming {out_path} ({len(results.get('cases', {}))} cases done)")
    else:
        results = {"meta": {}, "cases": {}}

    def _needs_work(case_id: str, group: str) -> bool:
        if case_id not in results.get("cases", {}):
            return True
        if args.rescore_failed:
            p = results["cases"][case_id].get("scores", {}).get(group, {}).get("parsed")
            return p is None or p.get("score") is None
        return False

    pending = [
        r for r in rows
        if any(_needs_work(r[0], g) for g in args.groups)
    ]
    if not pending:
        print("Nothing to score.")
        return

    print(f"Loading Gemma {args.gemma_size.upper()} + SAE ...")
    rt = load_gemma_scope(args.gemma_size)

    results["meta"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "gemma_model": rt.model_id,
        "gemma_size": args.gemma_size,
        "sae_release": rt.sae_release,
        "sae_id": rt.sae_id,
        "subject_model": args.subject_model,
        "groups": args.groups,
        "total_cases": len(rows),
        "rescore_failed": args.rescore_failed,
    }
    results.setdefault("cases", {})

    for case_id, case_summary, reasoning in pending:
        print(f"\n=== {case_id} ===")
        existing = results["cases"].get(case_id, {})
        if existing.get("sae") and args.rescore_failed:
            sae_info = existing["sae"]
        else:
            sae_info = extract_sae_summary(rt, reasoning, top_k=args.top_k)

        case_result = {
            "case_summary_preview": case_summary[:500],
            "reasoning_chars": len(reasoning),
            "sae": sae_info,
            "scores": dict(existing.get("scores", {})),
        }

        for group in args.groups:
            if not _needs_work(case_id, group):
                continue
            sae_block = sae_info["summary_text"] if group == "sae_augmented" else None
            prompt = build_scoring_prompt(case_summary, reasoning, group, sae_block)
            score_out = gemma_score(rt, prompt)
            case_result["scores"][group] = score_out
            parsed = score_out.get("parsed") or {}
            print(
                f"  [{group}] score={parsed.get('score', '?')} "
                f"rationale={(parsed.get('rationale') or score_out['raw_response'][:60])!r}"
            )

        results["cases"][case_id] = case_result
        _save(out_path, results)

    print(f"\nWrote {out_path} ({len(results['cases'])} cases)")


if __name__ == "__main__":
    main()
