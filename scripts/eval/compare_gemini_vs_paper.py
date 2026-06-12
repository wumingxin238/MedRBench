#!/usr/bin/env python3
"""Compare 35-case Gemini metrics (local judges) vs MedR-Bench paper (oracle setting)."""
import json
import glob
import os
from pathlib import Path

# Paper: Extended Table 3 (oracle accuracy, all diseases) & Table 4 (oracle reasoning)
PAPER_ORACLE = {
    "accuracy_pct": 86.83,
    "efficiency_pct": 95.89,
    "factuality_pct": 98.23,
    "completeness_pct": 83.28,  # called Completeness in paper; recall in code
}

PAPER_NOTE = (
    "Full benchmark (~491 oracle diagnosis cases), GPT-4o judge, WITH web search "
    "during factuality evaluation."
)


def mean_metric(files, key):
    vals = []
    for f in files:
        v = json.load(open(f, encoding="utf-8")).get(key)
        if v is not None:
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    root = Path(__file__).resolve().parents[2]
    configs = [
        ("Local GPT judge (reasoning_results/, likely with search)", "reasoning_results", "acc_results"),
        ("Local Qwen judge (reasoning_results_qwen_judge/, no search)", "reasoning_results_qwen_judge", "acc_results_qwen_judge"),
    ]
    print("=" * 72)
    print("Gemini-2.0-FT  |  Oracle diagnosis  |  35 cases (test_cases.json)")
    print("Paper reference:", PAPER_NOTE)
    print("=" * 72)
    print(f"{'Source':<42} {'Acc':>7} {'Eff':>7} {'Fact':>7} {'Comp':>7}")
    print("-" * 72)
    print(
        f"{'Paper (full set)':<42} "
        f"{PAPER_ORACLE['accuracy_pct']/100:>7.1%} "
        f"{PAPER_ORACLE['efficiency_pct']/100:>7.1%} "
        f"{PAPER_ORACLE['factuality_pct']/100:>7.1%} "
        f"{PAPER_ORACLE['completeness_pct']/100:>7.1%}"
    )
    for label, rdir, adir in configs:
        rfs = glob.glob(str(root / "src/Evaluation" / rdir / "gemini2-ft" / "*.json"))
        afs = glob.glob(str(root / "src/Evaluation" / adir / "gemini2-ft" / "*.json"))
        acc = sum(1 for f in afs if json.load(open(f, encoding="utf-8")).get("accuracy")) / len(afs) if afs else float("nan")
        print(
            f"{label[:42]:<42} "
            f"{acc:>7.1%} "
            f"{mean_metric(rfs,'efficiency'):>7.1%} "
            f"{mean_metric(rfs,'factulity'):>7.1%} "
            f"{mean_metric(rfs,'recall'):>7.1%}"
        )
    print("=" * 72)
    print("\nMetric mapping (code <-> paper):")
    print("  accuracy     <- oracle_diagnose_accuracy.py (outcome match)")
    print("  efficiency   <- reasoning_eval efficiency_score")
    print("  factuality   <- reasoning_eval factuality_score (paper: Factuality)")
    print("  completeness <- reasoning_eval recall_score (paper: Completeness)")
    print("\nInterpretation tips:")
    print("  - Accuracy on 35 cases is closest to paper (~86%).")
    print("  - Reasoning gaps are expected if: (1) Qwen vs GPT judge,")
    print("    (2) no web search, (3) 35-case subset != full 491 cases,")
    print("    (4) local gemini text may differ from HF oracle_diagnosis.json.")


if __name__ == "__main__":
    main()
