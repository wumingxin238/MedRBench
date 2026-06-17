#!/usr/bin/env python3
"""Run oracle_diagnose_accuracy for Stage-1 demo 100 × three subject models."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "src" / "Evaluation"
CASES = PROJECT_ROOT / "data" / "MedRBench" / "demo_diagnosis_100.json"
OUTPUTS = PROJECT_ROOT / "data" / "Stage1" / "oracle_diagnosis_subjects.json"
OUT = PROJECT_ROOT / "data" / "Stage1" / "acc_results"
MODELS = ["o3-mini", "deepseek-r1", "qwen3-8b"]


def main() -> None:
    for model in MODELS:
        out_dir = OUT / model
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(out_dir.glob("*.json")))
        print(f"\n==> {model} ({existing}/100 cached)")
        cmd = [
            sys.executable,
            "oracle_diagnose_accuracy.py",
            "--model",
            model,
            "--sequential",
            "--patient-cases",
            str(CASES),
            "--model-outputs",
            str(OUTPUTS),
            "--output-dir",
            str(OUT),
        ]
        subprocess.run(cmd, cwd=EVAL_DIR, check=True)


if __name__ == "__main__":
    main()
