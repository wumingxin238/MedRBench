#!/usr/bin/env python3
"""Compare Oracle 957 local Qwen-judge eval vs paper for all models."""
import json
import glob
import os
from pathlib import Path

MODELS = ["deepseek-r1", "gemini2-ft", "qwq"]

PAPER_ORACLE = {
    "gemini2-ft": {"acc": 86.83, "eff": 95.89, "fact": 98.23, "comp": 83.28},
    "qwq": {"acc": 85.06, "eff": 71.20, "fact": 84.02, "comp": 79.97},
    "deepseek-r1": {"acc": 89.76, "eff": 97.17, "fact": 95.03, "comp": 78.27},
}

PAPER_ORACLE_RARE = {
    "gemini2-ft": {"eff": 96.45, "fact": 98.39, "comp": 84.30},
    "qwq": {"eff": 72.25, "fact": 84.30, "comp": 80.70},
    "deepseek-r1": {"eff": 97.52, "fact": 95.18, "comp": 79.01},
}


def agg(reason_dir, acc_dir=None):
    fs = glob.glob(str(reason_dir / "PMC*.json"))
    if not fs:
        return None
    gt_path = (
        Path(__file__).resolve().parents[2]
        / "data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
    )
    gt = json.load(open(gt_path, encoding="utf-8"))
    rare = {cid for cid, c in gt.items() if c.get("checked_rare_disease")}

    buckets = {k: [] for k in ["eff", "fact", "comp"]}
    buckets_r = {k: [] for k in ["eff", "fact", "comp"]}
    steps, reasoning_steps, comp0 = [], [], 0

    for f in fs:
        d = json.load(open(f, encoding="utf-8"))
        cid = d.get("id", os.path.basename(f).replace(".json", ""))
        for code, key in [("eff", "efficiency"), ("fact", "factulity"), ("comp", "recall")]:
            v = d.get(key)
            if v is not None:
                buckets[code].append(float(v))
                if cid in rare:
                    buckets_r[code].append(float(v))
        if float(d.get("recall") or 0) < 0.01:
            comp0 += 1
        re = d.get("reasoning_eval", [])
        steps.append(len(re))
        reasoning_steps.append(sum(1 for s in re if s.get("efficiency") == "Reasoning"))

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    out = {
        "n": len(fs),
        "eff": mean(buckets["eff"]),
        "fact": mean(buckets["fact"]),
        "comp": mean(buckets["comp"]),
        "eff_r": mean(buckets_r["eff"]),
        "fact_r": mean(buckets_r["fact"]),
        "comp_r": mean(buckets_r["comp"]),
        "avg_steps": mean(steps),
        "avg_reasoning_steps": mean(reasoning_steps),
        "comp_zero_n": comp0,
    }
    if acc_dir and acc_dir.exists():
        afs = glob.glob(str(acc_dir / "PMC*.json"))
        if afs:
            out["acc"] = sum(
                1 for f in afs if json.load(open(f, encoding="utf-8")).get("accuracy")
            ) / len(afs)
            out["acc_n"] = len(afs)
        else:
            out["acc"] = None
    else:
        out["acc"] = None
    return out


def comp_distribution(ev, model):
    rec = [
        json.load(open(f, encoding="utf-8"))["recall"]
        for f in glob.glob(str(ev / f"reasoning_results_qwen_judge_paper_957/{model}/PMC*.json"))
    ]
    if not rec:
        return None
    bins = []
    for lo, hi, lb in [(0, 0.5, "0-50%"), (0.5, 0.7, "50-70%"), (0.7, 0.9, "70-90%"), (0.9, 1.01, "90-100%")]:
        c = sum(1 for x in rec if lo <= x < hi)
        bins.append((lb, c, 100 * c / len(rec)))
    return bins


def fmt_acc(a):
    if a is None:
        return "  N/A"
    return f"{100*a:6.2f}%"


def main():
    root = Path(__file__).resolve().parents[2]
    ev = root / "src/Evaluation"
    base_r = ev / "reasoning_results_qwen_judge_paper_957"
    base_a = ev / "acc_results_qwen_judge_paper_957"

    local = {
        m: agg(base_r / m, base_a / m if (base_a / m).exists() else None) for m in MODELS
    }

    print("=" * 82)
    print("Oracle 957 | Paper inference | Qwen2.5-7B Judge | NO web search")
    print("=" * 82)
    hdr = f"{'Model / Source':<26} {'Acc':>7} {'Eff':>7} {'Fact':>7} {'Comp':>7} {'n':>5}"
    print(hdr)
    print("-" * 82)

    for model in MODELS:
        p = PAPER_ORACLE[model]
        print(
            f"{model+' (paper GPT-4o)':<26} {p['acc']:>6.2f}% {p['eff']:>6.2f}% "
            f"{p['fact']:>6.2f}% {p['comp']:>6.2f}%"
        )
        a = local[model]
        if not a:
            print(f"{model+' (local Qwen)':<26} NO DATA")
            continue
        print(
            f"{model+' (local Qwen)':<26} {fmt_acc(a.get('acc')):>7} "
            f"{100*a['eff']:>6.2f}% {100*a['fact']:>6.2f}% {100*a['comp']:>6.2f}% {a['n']:>5}"
        )

    print("\n--- Local vs paper (Δ pp) ---")
    print(f"{'Model':<14} {'Acc':>8} {'Eff':>8} {'Fact':>8} {'Comp':>8}")
    for model in MODELS:
        a, p = local[model], PAPER_ORACLE[model]
        if not a:
            continue
        acc_d = "N/A" if a.get("acc") is None else f"{100*a['acc']-p['acc']:+.2f}"
        print(
            f"{model:<14} {acc_d:>8} {100*a['eff']-p['eff']:>+7.2f} "
            f"{100*a['fact']-p['fact']:>+7.2f} {100*a['comp']-p['comp']:>+7.2f}"
        )

    print("\n--- Same Judge ranking (local 957) ---")
    order = sorted(MODELS, key=lambda m: local[m]["comp"] if local[m] else -1, reverse=True)
    print("Completeness:", " > ".join(f"{m} {100*local[m]['comp']:.2f}%" for m in order))
    order = sorted(MODELS, key=lambda m: local[m]["eff"] if local[m] else -1, reverse=True)
    print("Efficiency:  ", " > ".join(f"{m} {100*local[m]['eff']:.2f}%" for m in order))
    order = sorted(MODELS, key=lambda m: local[m]["fact"] if local[m] else -1, reverse=True)
    print("Factuality:  ", " > ".join(f"{m} {100*local[m]['fact']:.2f}%" for m in order))
    if any(local[m].get("acc") is not None for m in MODELS):
        order = sorted(
            MODELS, key=lambda m: local[m].get("acc") or -1, reverse=True
        )
        print(
            "Accuracy:    ",
            " > ".join(
                f"{m} {100*local[m]['acc']:.2f}%"
                for m in order
                if local[m].get("acc") is not None
            ),
        )

    print("\n--- Paper ranking (GPT-4o) ---")
    print(
        "Accuracy:    ",
        " > ".join(
            f"{m} {PAPER_ORACLE[m]['acc']:.2f}%"
            for m in sorted(MODELS, key=lambda m: PAPER_ORACLE[m]["acc"], reverse=True)
        ),
    )
    print(
        "Completeness:",
        " > ".join(
            f"{m} {PAPER_ORACLE[m]['comp']:.2f}%"
            for m in sorted(MODELS, key=lambda m: PAPER_ORACLE[m]["comp"], reverse=True)
        ),
    )

    print("\n--- 491 rare subset (local vs paper) ---")
    for model in MODELS:
        a = local[model]
        pr = PAPER_ORACLE_RARE.get(model)
        if not a or not pr:
            continue
        print(f"  {model}:")
        print(
            f"    local  Eff {100*a['eff_r']:.2f}%  Fact {100*a['fact_r']:.2f}%  "
            f"Comp {100*a['comp_r']:.2f}%  (comp=0: {a['comp_zero_n']})"
        )
        print(
            f"    paper  Eff {pr['eff']:.2f}%  Fact {pr['fact']:.2f}%  Comp {pr['comp']:.2f}%"
        )

    print("\n--- Completeness distribution ---")
    for model in MODELS:
        dist = comp_distribution(ev, model)
        if not dist:
            continue
        print(f"  {model}:")
        for lb, c, pct in dist:
            print(f"    {lb}: {c} ({pct:.1f}%)")

    print("\n--- Step counts (local) ---")
    for model in MODELS:
        a = local[model]
        if a:
            print(
                f"  {model}: avg steps {a['avg_steps']:.2f}, "
                f"Reasoning steps {a['avg_reasoning_steps']:.2f}, comp=0 cases {a['comp_zero_n']}"
            )

    # Pairwise comp delta vs deepseek
    if local["deepseek-r1"] and local["gemini2-ft"] and local["qwq"]:
        print("\n--- Pairwise Completeness delta (local) ---")
        pairs = [
            ("deepseek-r1", "gemini2-ft"),
            ("deepseek-r1", "qwq"),
            ("qwq", "gemini2-ft"),
        ]
        for a, b in pairs:
            da = local[a]["comp"] - local[b]["comp"]
            print(f"  {a} vs {b}: {100*da:+.2f} pp ({a} higher)" if da >= 0 else f"  {a} vs {b}: {100*da:+.2f} pp ({b} higher)")


if __name__ == "__main__":
    main()
