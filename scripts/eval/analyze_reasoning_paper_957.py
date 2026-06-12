#!/usr/bin/env python3
"""Analyze reasoning_results_qwen_judge_paper_957 vs paper."""
import json
import glob
import os
import statistics
from pathlib import Path

PAPER = {
    "accuracy": 86.83,
    "efficiency": 95.89,
    "factuality": 98.23,
    "completeness": 83.28,
}

KEY_MAP = {
    "efficiency": "efficiency",
    "factuality": "factulity",
    "completeness": "recall",
}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    root = Path(__file__).resolve().parents[2]
    rdir = root / "src/Evaluation/reasoning_results_qwen_judge_paper_957/gemini2-ft"
    adir = root / "src/Evaluation/acc_results_qwen_judge_paper_957/gemini2-ft"
    gt_path = root / "data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
    err_log = root / "src/Evaluation/gemini2-ft_error.log"

    files = sorted(glob.glob(str(rdir / "PMC*.json")))
    gt = json.load(open(gt_path, encoding="utf-8"))
    rare = {cid for cid, c in gt.items() if c.get("checked_rare_disease")}

    issues = {
        "missing_efficiency": [],
        "missing_factulity": [],
        "missing_recall": [],
        "empty_reasoning_eval": [],
        "zero_steps": [],
        "all_steps_non_reasoning": [],
        "zero_recall_with_steps": [],
    }

    metrics = {k: [] for k in KEY_MAP}
    metrics_rare = {k: [] for k in KEY_MAP}
    metrics_nonrare = {k: [] for k in KEY_MAP}
    step_stats = {"total_steps": [], "reasoning_steps": [], "factual_steps": []}
    eff_cats = {}

    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        cid = d.get("id", os.path.basename(f).replace(".json", ""))
        for paper_k, code_k in KEY_MAP.items():
            v = d.get(code_k)
            if v is None:
                issues[f"missing_{code_k}"].append(cid)
            else:
                metrics[paper_k].append(float(v))
                bucket = metrics_rare if cid in rare else metrics_nonrare
                bucket[paper_k].append(float(v))

        re = d.get("reasoning_eval", [])
        if not re:
            issues["empty_reasoning_eval"].append(cid)
        else:
            n = len(re)
            rs = sum(1 for s in re if s.get("efficiency") == "Reasoning")
            fs = sum(1 for s in re if s.get("factulity") is True)
            step_stats["total_steps"].append(n)
            step_stats["reasoning_steps"].append(rs)
            step_stats["factual_steps"].append(fs)
            for s in re:
                cat = s.get("efficiency", "?")
                eff_cats[cat] = eff_cats.get(cat, 0) + 1
            if n == 0:
                issues["zero_steps"].append(cid)
            if rs == 0 and n > 0:
                issues["all_steps_non_reasoning"].append(cid)
            if d.get("recall") == 0 and n > 0:
                issues["zero_recall_with_steps"].append(cid)

    acc_files = glob.glob(str(adir / "PMC*.json"))
    acc_ok = sum(
        1 for f in acc_files if json.load(open(f, encoding="utf-8")).get("accuracy") is True
    )
    acc_rate = 100 * acc_ok / len(acc_files) if acc_files else float("nan")

    local957 = {
        "accuracy": acc_rate,
        "efficiency": 100 * mean(metrics["efficiency"]),
        "factuality": 100 * mean(metrics["factuality"]),
        "completeness": 100 * mean(metrics["completeness"]),
    }
    local491 = {
        "efficiency": 100 * mean(metrics_rare["efficiency"]),
        "factuality": 100 * mean(metrics_rare["factuality"]),
        "completeness": 100 * mean(metrics_rare["completeness"]),
    }

    print("=" * 72)
    print("Gemini-2.0-FT | Oracle reasoning | Qwen2.5-7B judge | NO web search")
    print("=" * 72)
    print(f"Reasoning files: {len(files)} / 957")
    print(f"Accuracy files:  {len(acc_files)} / 957")
    if err_log.exists():
        lines = err_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        print(f"Error log lines: {len(lines)}")
        if lines:
            print("  Last 3 errors:")
            for ln in lines[-3:]:
                print(f"    {ln[:120]}")

    print("\n--- Data quality ---")
    ok = True
    for k, v in issues.items():
        print(f"  {k}: {len(v)}")
        if v:
            ok = False
            if len(v) <= 3:
                print(f"    examples: {v}")
    if ok and len(files) == 957:
        print("  All 957 cases complete with required fields.")

    print("\n--- Step-level stats (aggregated) ---")
    for k, v in step_stats.items():
        print(f"  {k}: mean={mean(v):.2f}, median={statistics.median(v):.1f}")
    print("  efficiency category counts across all steps:")
    total_cats = sum(eff_cats.values())
    for cat, c in sorted(eff_cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {c} ({100*c/total_cats:.1f}%)")

    print("\n--- Metrics vs Paper (Extended Table 3/4, oracle, gemini-2.0-FT) ---")
    print(f"{'Metric':<14} {'Paper':>8} {'957 all':>10} {'491 rare':>10} {'Gap(957)':>10}")
    print("-" * 54)
    for k in ["accuracy", "efficiency", "factuality", "completeness"]:
        p = PAPER[k]
        l957 = local957[k]
        l491 = local491.get(k)
        if l491 is not None:
            print(f"{k:<14} {p:>7.2f}% {l957:>9.2f}% {l491:>9.2f}% {l957 - p:>+9.2f}pp")
        else:
            print(f"{k:<14} {p:>7.2f}% {l957:>9.2f}% {'—':>10} {l957 - p:>+9.2f}pp")

    rec = metrics["completeness"]
    print("\n--- Completeness (recall) distribution ---")
    for lo, hi, label in [(0, 0.5, "0-50%"), (0.5, 0.7, "50-70%"), (0.7, 0.9, "70-90%"), (0.9, 1.01, "90-100%")]:
        c = sum(1 for x in rec if lo <= x < hi)
        print(f"  {label}: {c} ({100 * c / len(rec):.1f}%)")

    low = sorted(
        [(json.load(open(f, encoding="utf-8"))["id"], json.load(open(f, encoding="utf-8"))["recall"]) for f in files],
        key=lambda x: x[1],
    )[:10]
    print("\n--- Lowest completeness cases ---")
    for cid, r in low:
        print(f"  {cid}: {r * 100:.1f}%")

    # 35 subset if acc exists
    test_path = root / "data/MedRBench/test_cases.json"
    if test_path.exists():
        test_ids = set(json.load(open(test_path, encoding="utf-8")).keys())
        m35 = {k: [] for k in KEY_MAP}
        for f in files:
            cid = os.path.basename(f).replace(".json", "")
            if cid not in test_ids:
                continue
            d = json.load(open(f, encoding="utf-8"))
            for paper_k, code_k in KEY_MAP.items():
                if d.get(code_k) is not None:
                    m35[paper_k].append(float(d[code_k]))
        print("\n--- 35 test_cases subset ---")
        for k in ["efficiency", "factuality", "completeness"]:
            print(f"  {k}: {100 * mean(m35[k]):.2f}% (n={len(m35[k])})")


if __name__ == "__main__":
    main()
