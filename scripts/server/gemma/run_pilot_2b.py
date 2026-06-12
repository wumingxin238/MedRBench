#!/usr/bin/env python3
"""
Stage-1 pilot: Gemma 2B + Gemma Scope on a few MedRBench cases.

Groups:
  direct         — score reasoning text only
  sae_augmented  — score reasoning + SAE feature summary

Example (from project root on server):
  conda activate gemma_scope
  cd ~/MedRBench
  python scripts/server/gemma/run_pilot_2b.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gemma_scope_utils import (  # noqa: E402
    build_scoring_prompt,
    extract_sae_summary,
    gemma_score,
    iter_pilot_cases,
    load_gemma_scope_2b,
)

PROJECT_ROOT = SCRIPT_DIR.parents[2]
# demo_oracle_diagnosis_strong aligns with demo_diagnosis_100 case IDs
DEFAULT_CASES = PROJECT_ROOT / "data" / "MedRBench" / "demo_diagnosis_100.json"
DEFAULT_OUTPUTS = PROJECT_ROOT / "data" / "MedRBench" / "demo_oracle_diagnosis_strong.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "src" / "Evaluation" / "gemma_scope_pilot_2b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--subject-model", default="deepseek-r1")
    parser.add_argument("--limit", type=int, default=3, help="Number of cases (pilot)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["direct", "sae_augmented"],
        choices=["direct", "sae_augmented"],
    )
    args = parser.parse_args()

    rows = iter_pilot_cases(
        str(args.cases),
        str(args.outputs),
        args.subject_model,
        limit=args.limit,
    )
    if not rows:
        raise SystemExit(
            f"No cases matched subject={args.subject_model!r}. "
            f"Check --cases and --outputs."
        )

    print(f"Loading Gemma 2B + SAE...")
    rt = load_gemma_scope_2b()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gemma_model": rt.model_id,
            "sae_release": rt.sae_release,
            "sae_id": rt.sae_id,
            "subject_model": args.subject_model,
            "groups": args.groups,
            "case_ids": [r[0] for r in rows],
        },
        "cases": {},
    }

    for case_id, case_summary, reasoning in rows:
        print(f"\n=== {case_id} ===")
        sae_info = extract_sae_summary(rt, reasoning, top_k=args.top_k)
        case_result = {
            "case_summary_preview": case_summary[:500],
            "reasoning_chars": len(reasoning),
            "sae": sae_info,
            "scores": {},
        }

        for group in args.groups:
            sae_block = sae_info["summary_text"] if group == "sae_augmented" else None
            prompt = build_scoring_prompt(case_summary, reasoning, group, sae_block)
            score_out = gemma_score(rt, prompt)
            case_result["scores"][group] = score_out
            parsed = score_out.get("parsed") or {}
            print(
                f"  [{group}] score={parsed.get('score', '?')} "
                f"rationale={parsed.get('rationale', score_out['raw_response'][:80])!r}"
            )

        results["cases"][case_id] = case_result

    out_path = args.out_dir / f"pilot_{args.subject_model}_n{len(rows)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
