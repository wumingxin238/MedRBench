#!/usr/bin/env python3
"""Direct vs Aug reasoning stats stratified by Gemma Acc correct/wrong."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Stage2"
MODELS = ["deepseek-r1", "qwen3-14b-thinking", "o3-mini"]


def load_acc(model: str) -> dict[str, bool]:
    d = DATA / f"acc_results_gemma/{model}"
    return {f.stem: bool(json.loads(f.read_text(encoding="utf-8")).get("accuracy")) for f in d.glob("*.json")}


def load_re(model: str, group: str) -> dict[str, dict]:
    p = DATA / f"reasoning_eval/diagnosis_gemma-9b-it_{model}_{group}.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {cid: r for cid, r in doc["cases"].items() if r.get("status") == "ok"}


def mean_metric(rows: list[dict], key: str) -> float:
    return st.mean(r[key] for r in rows) * 100 if rows else float("nan")


def row(model: str, ids: list[str], acc: dict[str, bool]) -> dict:
    direct = load_re(model, "direct")
    aug = load_re(model, "inference_augmented")
    dr = [direct[i] for i in ids]
    ar = [aug[i] for i in ids]
    n = len(ids)
    de = mean_metric(dr, "efficiency")
    ae = mean_metric(ar, "efficiency")
    df = mean_metric(dr, "factulity")
    af = mean_metric(ar, "factulity")
    drec = mean_metric(dr, "recall")
    arec = mean_metric(ar, "recall")
    return {
        "model": model,
        "n": n,
        "acc_pct": 100 * sum(acc[i] for i in ids) / n if n else 0.0,
        "direct_eff": de,
        "aug_eff": ae,
        "direct_fact": df,
        "aug_fact": af,
        "direct_rec": drec,
        "aug_rec": arec,
        "delta_eff": ae - de,
        "delta_rec": arec - drec,
        "delta_fact": af - df,
    }


def main() -> None:
    splits = {
        "all": lambda acc: True,
        "correct": lambda acc: acc,
        "wrong": lambda acc: not acc,
    }
    out: dict[str, list[dict]] = {k: [] for k in splits}

    for model in MODELS:
        acc = load_acc(model)
        direct = load_re(model, "direct")
        aug = load_re(model, "inference_augmented")
        ids = sorted(set(acc) & set(direct) & set(aug))
        for name, fn in splits.items():
            sub = [i for i in ids if fn(acc[i])]
            out[name].append(row(model, sub, acc))

    for name, rows in out.items():
        print(f"\n=== {name.upper()} (n per model) ===")
        for r in rows:
            print(
                f"{r['model']:22} n={r['n']:3d} acc={r['acc_pct']:5.1f}% "
                f"D-Eff={r['direct_eff']:.1f} A-Eff={r['aug_eff']:.1f} "
                f"D-Fact={r['direct_fact']:.1f} A-Fact={r['aug_fact']:.1f} "
                f"D-Rec={r['direct_rec']:.1f} A-Rec={r['aug_rec']:.1f} "
                f"dEff={r['delta_eff']:+.1f} dFact={r['delta_fact']:+.1f} dRec={r['delta_rec']:+.1f}"
            )

    # markdown tables
    print("\n\n--- MARKDOWN ---\n")
    titles = {
        "all": "400 例全体：Direct vs Aug（Gemma Acc 分层基准）",
        "correct": "Acc 正确子集：Direct vs Aug",
        "wrong": "Acc 错误子集：Direct vs Aug",
    }
    for name, rows in out.items():
        print(f"### {titles[name]}\n")
        print("| 模型 | n | Gemma Acc | Direct Eff | Aug Eff | Direct Fact | Aug Fact | Direct Rec | Aug Rec | Δ Eff | Δ Rec |")
        print("|------|---|-----------|------------|---------|-------------|----------|------------|---------|-------|-------|")
        for r in rows:
            print(
                f"| {r['model']} | {r['n']} | {r['acc_pct']:.1f}% "
                f"| {r['direct_eff']:.1f}% | {r['aug_eff']:.1f}% "
                f"| {r['direct_fact']:.1f}% | {r['aug_fact']:.1f}% "
                f"| {r['direct_rec']:.1f}% | {r['aug_rec']:.1f}% "
                f"| {r['delta_eff']:+.1f} pp | {r['delta_rec']:+.1f} pp |"
            )
        print()


if __name__ == "__main__":
    main()
