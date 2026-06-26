#!/usr/bin/env python3
"""Stage-2 aggregate analysis for all subject models."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ["qwen3-14b-thinking", "o3-mini", "deepseek-r1"]


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def mean(vals: list[float]) -> float:
    return st.mean(vals) if vals else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def load_acc(model: str, source: str = "gpt") -> dict[str, bool]:
    sub = "acc_results_gpt" if source == "gpt" else "acc_results_gemma"
    d = DATA / f"Stage2/{sub}/{model}"
    return {f.stem: bool(load(f).get("accuracy")) for f in d.glob("*.json")}


def load_re(model: str, group: str) -> tuple[dict, dict]:
    p = DATA / f"Stage2/reasoning_eval/diagnosis_gemma-9b-it_{model}_{group}.json"
    if not p.is_file():
        return {}, {}
    d = load(p)
    return d.get("cases", {}), d.get("meta", {})


def main() -> None:
    manifest = load(DATA / "MedRBench/stage2_manifest.json")
    demo = set(manifest["diagnosis"]["demo_case_ids"])
    hard = set(manifest["diagnosis"]["hard_case_ids"])
    all_ids = manifest["diagnosis"]["case_ids"]

    print("=" * 78)
    print("STAGE-2 ALL MODELS × 400 (100 demo + 300 hard)")
    print("=" * 78)

    rows_summary = []
    for model in MODELS:
        acc = load_acc(model, "gemma")
        acc_gpt = load_acc(model, "gpt")
        n_files = len(acc)
        tot = sum(acc.values())
        demo_ok = sum(1 for c in demo if acc.get(c))
        hard_ok = sum(1 for c in hard if acc.get(c))
        rows_summary.append((model, n_files, tot, demo_ok, hard_ok))

        print(f"\n{'#' * 78}")
        print(f"# {model}")
        print(f"{'#' * 78}")
        tot_gpt = sum(acc_gpt.values())
        print(f"\nAcc (Gemma-9B): files={n_files}/400  correct={tot}/400 = {100*tot/400:.1f}%")
        print(f"  Demo: {demo_ok}/100 = {demo_ok:.0f}%   Hard: {hard_ok}/300 = {100*hard_ok/300:.1f}%")
        print(f"Acc (GPT-4o):   correct={tot_gpt}/400 = {100*tot_gpt/400:.1f}%  (delta {100*(tot-tot_gpt)/400:+.1f} pp)")

        for group in ("direct", "inference_augmented"):
            cases, meta = load_re(model, group)
            ok_n = sum(1 for c in cases.values() if c.get("status") == "ok")
            print(
                f"\nGemma-9B {group}: {ok_n}/400 ok"
                f"  Eff={meta.get('mean_efficiency', 0)*100:.1f}%"
                f"  Fact={meta.get('mean_factulity', 0)*100:.1f}%"
                f"  Rec={meta.get('mean_recall', 0)*100:.1f}%"
            )
            for label, ids in [("Demo", demo), ("Hard", hard)]:
                rs = [cases[c] for c in ids if cases.get(c, {}).get("status") == "ok"]
                if not rs:
                    continue
                print(
                    f"  {label}: Eff={mean([r['efficiency'] for r in rs])*100:.1f}%"
                    f" Fact={mean([float(r['factulity']) for r in rs])*100:.1f}%"
                    f" Rec={mean([r['recall'] for r in rs])*100:.1f}%"
                    f" | Acc={sum(1 for c in ids if acc.get(c))}/{len(ids)}"
                )

        direct, _ = load_re(model, "direct")
        aug, _ = load_re(model, "inference_augmented")
        de, dr, ae, ar = [], [], [], []
        for cid in all_ids:
            d, a = direct.get(cid), aug.get(cid)
            if d and a and d.get("status") == "ok" and a.get("status") == "ok":
                de.append(d["efficiency"])
                dr.append(d["recall"])
                ae.append(a["efficiency"])
                ar.append(a["recall"])
        if de:
            print(
                f"\naug - direct: Eff {100*(mean(ae)-mean(de)):+.1f} pp"
                f"  Rec {100*(mean(ar)-mean(dr)):+.1f} pp"
            )

        for group in ("direct", "inference_augmented"):
            cases, _ = load_re(model, group)
            correct = [cases[c] for c in all_ids if acc.get(c) and cases.get(c, {}).get("status") == "ok"]
            wrong = [
                cases[c]
                for c in all_ids
                if c in acc and not acc[c] and cases.get(c, {}).get("status") == "ok"
            ]
            if not correct or not wrong:
                continue
            dc = mean([x["recall"] for x in correct]) - mean([x["recall"] for x in wrong])
            de2 = mean([x["efficiency"] for x in correct]) - mean([x["efficiency"] for x in wrong])
            r = pearson(
                [1.0 if acc[c] else 0.0 for c in all_ids if cases.get(c, {}).get("status") == "ok"],
                [cases[c]["recall"] for c in all_ids if cases.get(c, {}).get("status") == "ok"],
            )
            print(
                f"\nAcc×reasoning ({group}): Rec Δ(c-w)={100*dc:+.1f}pp"
                f" Eff Δ={100*de2:+.1f}pp  r(acc,Rec)={r:.2f}"
            )

    print(f"\n{'=' * 78}")
    print("SUMMARY TABLE")
    print(f"{'=' * 78}")
    print(f"{'Model':<22} {'Acc':>8} {'Demo':>8} {'Hard':>8} {'Eff_d':>7} {'Rec_d':>7} {'Eff_a':>7} {'Rec_a':>7}")
    print("-" * 78)
    for model, n_files, tot, demo_ok, hard_ok in rows_summary:
        _, md = load_re(model, "direct")
        _, ma = load_re(model, "inference_augmented")
        flag = "" if n_files >= 400 else f" [!{n_files}/400]"
        print(
            f"{model:<22} {100*tot/400:6.1f}%{flag:6} {demo_ok:>5}/100"
            f" {100*hard_ok/300:6.1f}%"
            f" {md.get('mean_efficiency', 0)*100:6.1f}%"
            f" {md.get('mean_recall', 0)*100:6.1f}%"
            f" {ma.get('mean_efficiency', 0)*100:6.1f}%"
            f" {ma.get('mean_recall', 0)*100:6.1f}%"
        )

    print("\nCompleteness check:")
    for model in MODELS:
        for src, label in (("gemma", "acc-gemma"), ("gpt", "acc-gpt")):
            acc_n = len(list((DATA / f"Stage2/acc_results_{src if src=='gpt' else 'gemma'}/{model}").glob("*.json")))
            print(f"  {model:22} {label}={acc_n}/400")


if __name__ == "__main__":
    main()
