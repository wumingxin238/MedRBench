#!/usr/bin/env python3
"""Stage-2: GPT judge (default gpt-5) diagnosis accuracy for qwen3-14b-thinking × 400 cases."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "src" / "Evaluation"
STAGE2 = PROJECT_ROOT / "data" / "Stage2"

DEFAULT_CASES = PROJECT_ROOT / "data" / "MedRBench" / "diagnosis_400.json"
DEFAULT_OUTPUTS = STAGE2 / "oracle_diagnosis_subjects.json"
DEFAULT_OUT = STAGE2 / "acc_results_gpt"
DEFAULT_MODEL = "qwen3-14b-thinking"
DEFAULT_JUDGE = os.environ.get("EVAL_MODEL", "gpt-5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-model", default=DEFAULT_MODEL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eval-model", default=DEFAULT_JUDGE, help="Judge model, e.g. gpt-5")
    parser.add_argument("--workers", type=int, default=8, help="Parallel API workers")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir / args.subject_model
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.cases.resolve()
    outputs_path = args.outputs.resolve()
    out_dir_root = args.out_dir.resolve()
    existing = len(list(out_dir.glob("*.json")))
    total = len(__import__("json").loads(cases_path.read_text(encoding="utf-8")))
    print(f"==> {args.subject_model} accuracy  judge={args.eval_model}  ({existing}/{total} cached)")

    cmd = [
        sys.executable,
        "oracle_diagnose_accuracy.py",
        "--model",
        args.subject_model,
        "--patient-cases",
        str(cases_path),
        "--model-outputs",
        str(outputs_path),
        "--output-dir",
        str(out_dir_root),
        "--eval-model",
        args.eval_model,
        "--workers",
        str(args.workers),
    ]
    if args.sequential:
        cmd.append("--sequential")
    subprocess.run(cmd, cwd=EVAL_DIR, check=True)


if __name__ == "__main__":
    main()
